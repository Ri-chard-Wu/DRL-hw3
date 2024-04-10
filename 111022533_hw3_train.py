import gym
import gym_multi_car_racing 
import numpy as np
from multiprocessing import Process, Pipe
from collections import deque

import tensorflow.keras.backend as K
import tensorflow as tf
import importlib

 
# env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
#         use_random_direction=True, backwards_flag=True, h_ratio=0.25,
#         use_ego_color=False)

# obs = env.reset()
# done = False
# total_reward = 0
 
# num_actions = env.action_space.shape[0]
# a_min = env.action_space.low
# a_max = env.action_space.high

# print(f'num_actions: {num_actions}, a_min: {a_min}, a_max: {a_max}')

# while not done: 
#   action = np.array([0, 1, 0] )
#   obs, reward, done, info = env.step(action)
#   total_reward += reward
# print("individual scores:", total_reward)


 


num_actions = 3
a_min = np.array([-1.0,  0.0,  0.0])
a_max = np.array([1.0, 1.0, 1.0])

 




para = AttrDict({
    
    'k': 4,
    'skip': 4,
    'img_shape': (84, 84),


    'lr': 1e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'ppo_clip': 0.2,
    'w_val': 0.5,
    'w_ent': 0.01,
    'horizon': 128,
    'epochs': 10,
    'batch_size': 128,
    'n_envs': 8,

    'save_period': 1000,
    'eval_period': 200,

    'ckpt_save_path': "ckpt/checkpoint0.h5",
    # 'ckpt_load_path': "ckpt/checkpoint1.h5"
})




 

class FrameSkipEnv:
    def __init__(self, env):   
        self.env = env 
        self.skip = para.skip
        
    def step(self, action):
        """
        - 4 steps at a time. 
        - take last obs, done and info, sum all 4 rewards.
        - clip reward between -1, 1.
        - return if encounter done before 4 steps.
        """        
        # self.t += 1

        cum_reward = 0

        for i in range(self.skip):
        
            obs, reward, done, info = self.env.step(action)  
            cum_reward += reward

            if done: break

        # cum_reward = min(max(cum_reward, -1), 1)
        return obs, cum_reward, done, info
 
    def reset(self):
        # self.t = 0
        # self.episode += 1
        return self.env.reset()




def make_env():
    env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
        use_random_direction=True, backwards_flag=True, h_ratio=0.25,
        use_ego_color=False)    
    env =  FrameSkipEnv(env)
    return env
 
 
def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'render':
            remote.send(env.render(mode='rgb_array'))
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class VecEnv():
 
    def __init__(self, env_fns):
        
        self.closed = False 
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(len(env_fns))])

        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]

        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        
    def step(self, actions): # actions: (n_env, n_actions) numpy array.

        # for remote, action in zip(self.remotes, actions):
        #     remote.send(('step', action))

        for i in range(len(self.remote)):
            self.remotes.send(('step', actions[i]))
        
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])
 
    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True








class Backbone(tf.keras.layers.Layer):

    def __init__(self):
        super(Backbone, self).__init__(name='backbone')

    def build(self, input_shape):

        self.seq = []
        self.seq.append(tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4))
        self.seq.append(tf.keras.layers.ReLU())
        self.seq.append(tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2))
        self.seq.append(tf.keras.layers.ReLU())
        self.seq.append(tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=1))
        self.seq.append(tf.keras.layers.ReLU())
  
       
    def call(self, x, training=False): # x.shape: (64, 65, 64) = (64, 65, hidden_size)
        for layer in self.seq: x = layer(x, training=training)
        return x



class Agent(tf.keras.Model):

    def __init__(self):  

        self.backbone = Backbone()
        self.flatten = tf.keras.layers.Flatten()
        self.a_mean_head = tf.keras.layers.Dense(units=num_actions, activation='tanh', name="a_mean_head")
        self.a_std = self.add_weight("a_std", shape=[num_actions,], trainable=False
                      initializer = tf.keras.initializers.Constant(value=0.5), dtype=tf.float32)
        
        self.v_head = tf.keras.layers.Dense(units=1, name="v_head")
        
    @tf.function
    def call(self, x, training=False): # (512, 224, 224, 3)
        x = self.backbone(x)
        x = self.flatten(x)
        return self.a_mean_head(x), self.v_head(x)


    def predict(self, state):
        
        state = tf.convert_to_tensor(state, tf.float32)         
        self.a_mean, self.v = self(state, True) # (b, num_actions), (b, 1)

        self.a_mean = a_min + ((self.a_mean + 1) / 2) * (a_max - a_min)

        self.dist = tf.compat.v1.distributions.Normal(self.a_mean, self.a_std, validate_args=True)
        self.a = tf.squeeze(self.dist.sample(1)) # (b, num_actions)        
        self.a_logP = tf.reduce_sum(self.dist.log_prob(self.a), axis=-1) # (b,)

        return self.a.numpy(), self.a_logP.numpy(), tf.squeeze(self.v, axis=-1).numpy()





 
def preprocess_frame(img):     
    img = img[:-12, 6:-6] # (84, 84, 3)
    img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    img = img / 255.0
    img = img * 2 - 1     
    assert img.shape == (84, 84)
    return img




class VecBuf():

    def __init__(self, n_envs):        

       
        self.n_envs = n_envs
        self.n = 0
         
        self.stack = [deque(maxlen=para.k) for _ in range(para.n_envs)]
        self.add_frame([np.zeros((*para.img_shape, 3)) for _ in n_envs])
         
        self.sta = np.zeros((para.horizon+1, n_envs, *para.img_shape, para.k))
        self.val = np.zeros((para.horizon+1, n_envs)) 
        self.act = np.zeros((para.horizon, n_envs, num_actions))               
        self.alg = np.zeros((para.horizon, n_envs))               
        self.rew = np.zeros((para.horizon, n_envs))
        self.don = np.zeros((para.horizon, n_envs), dtype='bool')
        self.adv = None
        self.ret = None

    

    def add_frame(self, frames):
        
        for i in range(self.n_envs):
            self.stack[i].append(preprocess_frame(frames[i]))
            self.sta[self.n, i] = np.stack(self.stack[i], axis=-1)
        self.n+=1
        # return [sta for sta in self.sta[self.n, :]]
        return self.sta[self.n, :] # (n_env, 84, 84, 4)
 
    def add_value(self, val):
        self.val[self.n-1] = val # (n_env,)

    def add_effects(self, act, alg, rew, don):
        idx = self.n-1
        self.act[idx] = act
        self.alg[idx] = alg
        self.rew[idx] = rew
        self.don[idx] = don
     
    def add_advantage(self, advs):
        self.adv = np.array(advs)
        assert self.adv.shape == self.val[:-1].shape
        self.ret = self.adv + self.val[:-1]
    
    def get_data(self):
    
        return AttrDict({
                'sta': self.sta,
                'act': self.act,
                'alg': self.alg,
                'val': self.val,
                'rew': self.rew,
                'don': self.don.astype(np.int32),
                'adv': self.adv,
                'ret': self.ret
            })

    def flatten(self):
        self.sta = self.sta.reshape((-1, *para.img_shape, para.k))       # [T x N, 84, 84, 4]
        self.act = self.act.reshape((-1, num_actions))  # [T x N, 3]        
        self.alg = self.alg.flatten() 
        self.val = self.val.flatten()  
        self.ret = self.ret.flatten()        
        self.adv = self.adv.flatten()

    def shuffle(self):
        self.idxes = np.arange(para.horizon * self.n_envs)
        np.random.shuffle(self.idxes)

    def batch(self):

        n = int(np.ceil(para.horizon * self.n_envs / para.batch_size))
    
        for i in range(n):
            
            idxes = self.idxes[i * para.batch_size : (i+1) * para.batch_size]

            yield AttrDict({
                'sta': self.sta[idxes],
                'act': self.act[idxes],
                'alg': self.alg[idxes],
                'val': self.val[idxes],
                'ret': self.ret[idxes],
                'adv': self.adv[idxes]
            })
            
      
def compute_gae(rewards, values, masks, gamma, LAMBDA):

    def _compute_gae(ewards, values, masks):
        gae = 0
        adv = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + gamma * LAMBDA * masks[i] * gae
            adv.append(gae)

        adv.reverse()
        return adv   

    adv2D = []
    for i in range(para.n_envs):
        adv2D.append(_compute_gae(rewards[:,i,...], values[:,i,...], masks[:,i,...]))

    return np.array(adv2D)
 






class Trainer():

    def __init__(self):
        pass


    def train():   
 
        model = PPO()

        env = VecEnv([make_env for _ in range(para.n_envs)])
        buf = VecBuf(para.n_envs)

        obs = envs.reset()
        # envs.get_images()

        buf.add_frame(obs)
 
         
        while True:
         
            for _ in range(horizon): 
                sta = buf.add_frame(obs) 
                a, a_logP, value = model.predict(sta) 
                obs, reward, done, _ = env.step(a)
                buf.add_value(value)
                buf.add_effects(a, a_logP, reward, done)
                
            
            sta = buf.add_frame(obs)
            buf.add_value(model.predict(sta)[2])

            data = buf.get_data()
            advantages = compute_gae(data.rew, data.val, 1-data.don, para.gamma, para.gae_lambda)
            buf.add_advantage(advantages)



            advantages = compute_gae(rewards, values, dones, discount_factor, gae_lambda)

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  
            returns = advantages + values

            # Flatten arrays
            states = np.array(states).reshape((-1, *input_shape))       # [T x N, 84, 84, 4]
            taken_actions = np.array(taken_actions).reshape((-1, num_actions))  # [T x N, 3]            
            returns = returns.flatten() # [T x N]
            advantages = advantages.flatten() # [T x N]

            # T = len(rewards)
            # N = num_envs
            # assert states.shape == (T * N, input_shape[0], input_shape[1], frame_stack_size)
            # assert taken_actions.shape == (T * N, num_actions)
            # assert returns.shape == (T * N,)
            # assert advantages.shape == (T * N,)

            # Train for some number of epochs
            # model.update_old_policy()  # θ_old <- θ


            buf.flatten()

            for _ in range(num_epochs):
                # num_samples = len(states)
                # indices = np.arange(num_samples)
                # np.random.shuffle(indices)
                
                buf.shuffle() 
                
                for batch in buf.batch(): 

                    self.train_step(batch)
                

                    # model.train(states[mb_idx], taken_actions[mb_idx],
                    #             returns[mb_idx], advantages[mb_idx])

                    # # Evaluate model
                    # if model.step_idx % eval_interval == 0:
                    #     print("[INFO] Running evaluation...")

                    #     avg_reward, value_error = self.evaluate()

                    #     model.write_to_summary("eval_avg_reward", avg_reward)
                    #     model.write_to_summary("eval_value_error", value_error)

                    # # Save model
                    # if model.step_idx % save_interval == 0:
                    #     model.save()

    def train_step(self, batch):




    def evaluate(self):


        total_reward = 0

        test_env = gym.make(env_name)
        # test_env.seed(0)
        
        initial_frame = test_env.reset()
        
        frame_stack = FrameStack(
            initial_frame, stack_size=frame_stack_size,
            preprocess_fn=preprocess_frame)

        rendered_frame = test_env.render(mode="rgb_array")
        values, rewards, dones = [], [], []
        if make_video:
            video_writer = cv2.VideoWriter(os.path.join(model.video_dir, "step{}.avi".format(model.step_idx)),
                                        cv2.VideoWriter_fourcc(*"MPEG"), 30,
                                        (rendered_frame.shape[1], rendered_frame.shape[0]))
        while True:
            # Predict action given state: π(a_t | s_t; θ)
            state = frame_stack.get_state()
            action, value = model.predict(
                np.expand_dims(state, axis=0), greedy=False)
            frame, reward, done, _ = test_env.step(action[0])
            rendered_frame = test_env.render(mode="rgb_array")
            total_reward += reward
            dones.append(done)
            values.append(value)
            rewards.append(reward)
            frame_stack.add_frame(frame)
            if make_video:
                video_writer.write(cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR))
            if done:
                break
        if make_video:
            video_writer.release()
        returns = compute_returns(np.transpose([rewards], [1, 0]), [
                                0], np.transpose([dones], [1, 0]), discount_factor)
        value_error = np.mean(np.square(np.array(values) - returns))
        return total_reward, value_error



trainer = Trainer()
trainer.train()