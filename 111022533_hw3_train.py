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

 
  

class AttrDict(dict):
    def __getattr__(self, a):
        return self[a]
 




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
    'n_iters': int(1e7),
    'batch_size': 128,
    'n_envs': 8,

    'save_period': 50,    
    'log_period': 10,

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
            ob = np.squeeze(ob)

            # print(f'ob.shape: {ob.shape}, done: {done}')
            # exit()

            remote.send((ob, reward[0], done, info))
        elif cmd == 'reset':
            ob = env.reset()
            ob = np.squeeze(ob)

            # print(f'ob.shape: {np.squeeze(ob).shape}')
            # exit()           

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

        for i in range(len(self.remotes)):
            self.remotes[i].send(('step', actions[i]))
        
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)

        # print(f'np.stack(rews).shape: {np.stack(rews).shape}')
        # exit()

        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))

        obs = np.stack([remote.recv() for remote in self.remotes])

        # print(f'obs.shape: {obs.shape}')
        # exit()

        return obs
 
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

        super().__init__()

        self.backbone = Backbone()
        self.flatten = tf.keras.layers.Flatten()
        self.a_mean_head = tf.keras.layers.Dense(units=num_actions, activation='tanh', name="a_mean_head")
        self.a_std = self.add_weight("a_std", shape=[num_actions,], trainable=False,
                      initializer = tf.keras.initializers.Constant(value=0.5), dtype=tf.float32)
        
        self.v_head = tf.keras.layers.Dense(units=1, name="v_head")
        self.update_counts = 0
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=para.lr)

   
    @tf.function
    def call(self, x, training=False):
        
        x = self.backbone(x)
        x = self.flatten(x)
        a_mean, v = self.a_mean_head(x), self.v_head(x)

        a_mean = a_min + ((a_mean + 1) / 2) * (a_max - a_min)

        # self.dist = tf.compat.v1.distributions.Normal(a_mean, self.a_std, validate_args=True)


        return a_mean, tf.squeeze(v, axis=-1)

     

    def predict(self, state):
        
        state = tf.convert_to_tensor(state, tf.float32)         
        # a, a_logP, ent, v = self(state)
        a_mean, v = self(state)

        dist = tf.compat.v1.distributions.Normal(a_mean, self.a_std, validate_args=True)

        a = tf.squeeze(dist.sample(1)) # (b, num_actions)        
        a_logP = tf.reduce_sum(dist.log_prob(a), axis=-1) # (b,) 
                
        return a.numpy(), a_logP.numpy(), v.numpy()
 


    def train_step(self, batch):
        self.update_counts += 1
        loss = self._train_step(batch)
        return loss.numpy()

    @tf.function
    def _train_step(self, batch):
        
        sta, a, a_logP_old, val_old, ret, adv = batch

        with tf.GradientTape() as tape:

            # sta = tf.convert_to_tensor(self.sta[idxes], tf.float32) # (b, 84, 84, 4)
            # act = tf.convert_to_tensor(self.act[idxes], tf.float32) # (b, 3)
            # alg = tf.convert_to_tensor(self.alg[idxes], tf.float32) # (b,)
            # val = tf.convert_to_tensor(self.val[idxes], tf.float32) # (b,)
            # ret = tf.convert_to_tensor(self.ret[idxes], tf.float32) # (b,)
            # adv = tf.convert_to_tensor(self.adv[idxes], tf.float32) # (b,)

            eps = para.ppo_clip
 
            a_mean, val = self(sta, True)

            dist = tf.compat.v1.distributions.Normal(a_mean, self.a_std, validate_args=True)

            a_logP = tf.reduce_sum(dist.log_prob(a), axis=-1) # (b,) 
            ent = dist.entropy()       

            val_clip = val + tf.clip_by_value(val - val_old, 1-eps, 1+eps)
            val_loss1 = tf.square(ret - val) # (b,)
            val_loss2 = tf.square(ret - val_clip) # (b,)
            val_loss = tf.reduce_mean(tf.maximum(val_loss1, val_loss2))             

            r = tf.exp(a_logP - a_logP_old) # (b,)            
            pg_loss = - tf.reduce_mean(tf.minimum(r * adv, tf.clip_by_value(r, 1-eps, 1+eps) * adv))

            ent_loss = - tf.reduce_mean(tf.reduce_sum(ent, axis=-1))

            total_loss = pg_loss + para.w_val * val_loss + para.w_ent * ent_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    
        return total_loss


    def save_checkpoint(self, path):  
        print(f'- saved ckpt {path}') 
        self.save_weights(path)
         

    def load_checkpoint(self, path):         
        print(f'- loaded ckpt {path}') 
        self(tf.random.uniform(shape=[1, *para.img_shape, para.k]))
        self.load_weights(path)






def preprocess_frame(img):     
    img = img[:-12, 6:-6] # (84, 84, 3)
    img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    img = img / 255.0
    img = img * 2 - 1    
    # print(f'img.shape: {img.shape}') 
    assert img.shape == (84, 84)
    return img



class VecBuf():

    def __init__(self, n_envs):   
       
        self.n_envs = n_envs
        self.n = 0 
         
        self.sta = np.zeros((para.horizon+1, n_envs, *para.img_shape, para.k))
        self.val = np.zeros((para.horizon+1, n_envs)) 
        self.act = np.zeros((para.horizon, n_envs, num_actions))               
        self.alg = np.zeros((para.horizon, n_envs))               
        self.rew = np.zeros((para.horizon, n_envs))
        self.don = np.zeros((para.horizon, n_envs), dtype='bool')
        self.adv = None
        self.ret = None

        self.stack = [deque(maxlen=para.k) for _ in range(para.n_envs)]
        self.init_stack(np.ones((n_envs), dtype='bool'))


    def init_stack(self, don):
        for j in range(self.n_envs):
            if(don[j]):
                for _ in range(para.k): 
                    self.stack[j].append(np.zeros(para.img_shape))
            # print(f'len(self.stack[{i}]): {len(self.stack[i])}')

    def add_frame(self, frames):
        
        for i in range(self.n_envs):
            # print(f'frames[i].shape: { frames[i].shape}')
            self.stack[i].append(preprocess_frame(frames[i]))

            # print(f'len(self.stack[i]): {len(self.stack[i])}')

            self.sta[self.n, i] = np.stack(self.stack[i], axis=-1)

        self.n+=1
        
        return self.sta[self.n-1, :] # (n_env, 84, 84, 4)
 
    def add_value(self, val):
        self.val[self.n-1] = val # (n_env,)

    def add_effects(self, act, alg, rew, don):
        idx = self.n-1
        self.act[idx] = act
        self.alg[idx] = alg
        self.rew[idx] = rew
        self.don[idx] = don

        self.init_stack(don)
     
    def add_advantage(self, adv):
        
        assert adv.shape == self.val[:-1].shape

        self.ret = adv + self.val[:-1]
        self.adv = (adv - adv.mean()) / (adv.std() + 1e-8)  
    
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
        self.sta = self.sta.reshape((-1, *para.img_shape, para.k))  # (H * n_env, 84, 84, 4)
        self.act = self.act.reshape((-1, num_actions))              # (H * n_env, 3)
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

            sta = tf.convert_to_tensor(self.sta[idxes], tf.float32) # (b, 84, 84, 4)
            act = tf.convert_to_tensor(self.act[idxes], tf.float32) # (b, 3)
            alg = tf.convert_to_tensor(self.alg[idxes], tf.float32) # (b,)
            val = tf.convert_to_tensor(self.val[idxes], tf.float32) # (b,)
            ret = tf.convert_to_tensor(self.ret[idxes], tf.float32) # (b,)
            adv = tf.convert_to_tensor(self.adv[idxes], tf.float32) # (b,)
            yield sta, act, alg, val, ret, adv
            
      
def compute_gae(rews, vals, masks, gamma, LAMBDA):
    
    assert len(rews) == para.horizon


    adv = np.zeros((para.horizon, para.n_envs))

    for j in range(para.n_envs):

        rew, val, mask = rews[:,j], vals[:,j], masks[:,j]

        gae = 0 
        for i in reversed(range(para.horizon)):
            delta = rew[i] + gamma * val[i + 1] * mask[i] - val[i]
            gae = delta + gamma * LAMBDA * mask[i] * gae            
            adv[i, j] = gae

    return adv # (horizon, n_envs)
 






class Trainer():

    def __init__(self):

        self.agent = Agent()
        if('ckpt_load_path' in para): 
            self.agent.load_checkpoint(para.ckpt_load_path)



    def train(self):    

        env = VecEnv([make_env for _ in range(para.n_envs)])
        obs = env.reset()
    
 
        with open("log.txt", "w") as f: f.write("")
        log = {}

        for t in range(para.n_iters):

            buf = VecBuf(para.n_envs)
         
            for _ in range(para.horizon): 
                sta = buf.add_frame(obs) 
                # print(f'buf.n: {buf.n}')
                a, a_logP, value = self.agent.predict(sta) 
                obs, reward, done, _ = env.step(a)
                buf.add_value(value)
                buf.add_effects(a, a_logP, reward, done)
             
            sta = buf.add_frame(obs) 
            buf.add_value(self.agent.predict(sta)[2])

            data = buf.get_data()
            advantages = compute_gae(data.rew, data.val, 1-data.don, para.gamma, para.gae_lambda)
            buf.add_advantage(advantages)
            

            buf.flatten()
            losses = []
            for _ in range(para.epochs):
                buf.shuffle()                 
                for batch in buf.batch(): 
                    losses.append(self.agent.train_step(batch))


            if t % para.save_period == 0:
                self.agent.save_checkpoint(f"ckpt/checkpoint{t}.h5")
                
            
            if t % para.log_period == 0:
                log['cum_reward_mean'], log['cum_reward_std'] = \
                                        self.compute_cum_reward(data.rew, data.don)
                log['loss'] = np.mean(losses)
                with open("log.txt", "a") as f: f.write(f't: {t}, ' + str(log) + '\n')

        

    def compute_cum_reward(self, rew, don): 

        cum_rewards = []
        traj_lens = []

        cum_reward = [0] * para.n_envs
        traj_len = [0] * para.n_envs
        
        for i in range(para.horizon):
            for j in range(para.n_envs):

                if don[i, j]:

                    cum_rewards.append(cum_reward[j] + rew[i, j])
                    traj_lens.append(traj_len[j] + 1)

                    cum_reward[j] = 0
                    traj_len[j] = 0

                else:
                    cum_reward[j] += rew[i, j]
                    traj_len[j] += 1

        return np.mean(cum_rewards), np.std(cum_rewards)



trainer = Trainer()
trainer.train()