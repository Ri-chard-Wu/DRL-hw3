import gym
import gym_multi_car_racing 
import numpy as np
from multiprocessing import Process, Pipe

 
# env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
#         use_random_direction=True, backwards_flag=True, h_ratio=0.25,
#         use_ego_color=False)

# obs = env.reset()
# done = False
# total_reward = 0
 
# num_actions = env.action_space.shape[0]
# action_min = env.action_space.low
# action_max = env.action_space.high

# print(f'num_actions: {num_actions}, action_min: {action_min}, action_max: {action_max}')

# while not done: 
#   action = [0, 1, 0] 
#   obs, reward, done, info = env.step(action)
#   total_reward += reward
# print("individual scores:", total_reward)


 


num_actions = 3
action_min = [-1.0,  0.0,  0.0]
action_max = [1.0, 1.0, 1.0]

 




para = AttrDict({
    
    'k': 4,
    'skip': 4,
    'frame_shape': (84, 84, 1),


    'lr': 1e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'ppo_clip': 0.2,
    'w_val': 0.5,
    'w_ent': 0.01,
    'horizon': 128,
    'epochs': 10,
    'bact_size': 128,
    'envs': 8,

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

        
    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        
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






 
def preprocess_frame(img):     
    img = img[:-12, 6:-6] # (84, 84, 3)
    img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    # img = img / 255.0
    # img = img * 2 - 1    
    img = img[..., np.newaxis] # (84, 84, 1)
    assert img.shape == (84, 84, 1)
    return img




class VecBuf():

    def __init__(self, n_envs):        

        self.trainer = trainer

        self.n = 0
        self.n_envs = n_envs
        
        steps = para.horizon + para.k - 1    

        self.obs = np.zeros((steps, n_envs, *para.frame_shape), dtype=np.uint8)
        self.action = np.zeros((steps, n_envs, num_actions))
        self.value = np.zeros((steps, n_envs))        
        self.reward = np.zeros((steps, n_envs))
        self.done = np.zeros((steps, n_envs))

    

    def add_frame(self, frames):
        
        for i, frame in enumerate(frames):
            self.obs[self.n, i] = preprocess_frame(frame)

        self.n +=  1
        return i 

    def get_state(self, step):
        return 

    def _get_state(self, step, envId):    

        d = para.k - 1  
        idx = d + step
        
        _start = idx-d
        end = idx+1 # non-inclusive
        start = _start
        for i in range(_start, end-1):
            if self.done[i, envId] > 0.5: start = i+1
    
        n = para.k - (end - start)

        out = np.concatenate([np.zeros(para.frame_shape)[np.newaxis,...]]*n  +\
                         [obs[np.newaxis,...] for obs in self.obs[start:end, envId]], axis=3) / 255.0

        assert out.shape == (1, para.frame_shape[0], para.frame_shape[1], para.k)
        return out

        
    def add_effects(self, idx, action, reward, done):
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = int(done)
     
  

    def sample(self, size):     

        assert self.n >= size
 
        idxes = np.random.choice(np.arange(para.k-1, self.n-1), size=size, replace=False)   
 
        return self.retrive_data(idxes)



    def retrive_data(self, idxes):
        # print(f'idxes: {idxes}')
        state = np.concatenate([self.stack_frame(idx) for idx in idxes], axis=0) 
        action = np.array([self.action[idx] for idx in idxes])
        reward = np.array([self.reward[idx] for idx in idxes])
        state_next = np.concatenate([self.stack_frame(idx+1) for idx in idxes], axis=0) 
        done = np.array([self.done[idx] for idx in idxes])
        
        # print(f'max(state): {np.max(state)}, state: {state}')

        state = tf.convert_to_tensor(state, tf.float32) 
        action = tf.convert_to_tensor(action, tf.int32)
        reward = tf.convert_to_tensor(reward, tf.float32)
        state_next = tf.convert_to_tensor(state_next, tf.float32)   
        done = tf.convert_to_tensor(done, tf.float32)

        return state, action, reward, state_next, done







class Trainer():

    def __init__(self):
        pass


    def train():   
 
        model = PPO()

        env = VecEnv([make_env for _ in range(para.envs)])
        buf = VecBuf(para.envs)

        obs = envs.reset()
        # envs.get_images()

        buf.add_frame(obs)

        # frame_stacks = [FrameStack(initial_frames[i], stack_size=frame_stack_size,
        #                         preprocess_fn=preprocess_frame) for i in range(num_envs)]

         
        while True:
            # While there are running environments
            states, taken_actions, values, rewards, dones = [], [], [], [], []

            # Simulate game for some number of steps
            for _ in range(horizon):
                # Predict and value action given state
                # π(a_t | s_t; θ_old)
                states_t = [frame_stacks[i].get_state() for i in range(num_envs)]
                actions_t, values_t = model.predict(states_t)

                # Sample action from a Gaussian distribution
                envs.step_async(actions_t)
                frames, rewards_t, dones_t, _ = envs.step_wait()
                envs.get_images()  # render

                # Store state, action and reward
                # [T, N, 84, 84, 4]
                states.append(states_t)
                taken_actions.append(actions_t)              # [T, N, 3]
                values.append(np.squeeze(values_t, axis=-1))  # [T, N]
                rewards.append(rewards_t)                    # [T, N]
                dones.append(dones_t)                        # [T, N]

                # Get new state
                for i in range(num_envs):
                    # Reset environment's frame stack if done
                    if dones_t[i]:
                        for _ in range(frame_stack_size):
                            frame_stacks[i].add_frame(frames[i])
                    else:
                        frame_stacks[i].add_frame(frames[i])

            # Calculate last values (bootstrap values)
            states_last = [frame_stacks[i].get_state()
                        for i in range(num_envs)]
            last_values = np.squeeze(model.predict(
                states_last)[1], axis=-1)  # [N]

            advantages = compute_gae(
                rewards, values, last_values, dones, discount_factor, gae_lambda)
            advantages = (advantages - advantages.mean()) / \
                (advantages.std() + 1e-8)  # Move down one line?
            returns = advantages + values
            # Flatten arrays
            states = np.array(states).reshape(
                (-1, *input_shape))       # [T x N, 84, 84, 4]
            taken_actions = np.array(taken_actions).reshape(
                (-1, num_actions))  # [T x N, 3]
            # [T x N]
            returns = returns.flatten()
            # [T X N]
            advantages = advantages.flatten()

            T = len(rewards)
            N = num_envs
            assert states.shape == (
                T * N, input_shape[0], input_shape[1], frame_stack_size)
            assert taken_actions.shape == (T * N, num_actions)
            assert returns.shape == (T * N,)
            assert advantages.shape == (T * N,)

            # Train for some number of epochs
            model.update_old_policy()  # θ_old <- θ
            for _ in range(num_epochs):
                num_samples = len(states)
                indices = np.arange(num_samples)
                np.random.shuffle(indices)
                for i in range(int(np.ceil(num_samples / batch_size))):
                    # Evaluate model
                    if model.step_idx % eval_interval == 0:
                        print("[INFO] Running evaluation...")

                        avg_reward, value_error = self.evaluate()

                        model.write_to_summary("eval_avg_reward", avg_reward)
                        model.write_to_summary("eval_value_error", value_error)

                    # Save model
                    if model.step_idx % save_interval == 0:
                        model.save()

                    # Sample mini-batch randomly
                    begin = i * batch_size
                    end = begin + batch_size
                    if end > num_samples:
                        end = None
                    mb_idx = indices[begin:end]

                    # Optimize network
                    model.train(states[mb_idx], taken_actions[mb_idx],
                                returns[mb_idx], advantages[mb_idx])



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