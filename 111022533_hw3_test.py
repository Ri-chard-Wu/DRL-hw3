import gym
import gym_multi_car_racing 
import numpy as np
from multiprocessing import Process, Pipe
from collections import deque

import tensorflow.keras.backend as K
import tensorflow as tf
import importlib
import shutil
import cv2
import os
from PIL import Image 
 


num_actions = 3
a_min = np.array([-1.0,  0.0,  0.0])
a_max = np.array([1.0, 1.0, 1.0])

 
  

class AttrDict(dict):
    def __getattr__(self, a):
        return self[a]
 




para = AttrDict({
    
    'k': 4,
    'skip': 2,
    'img_shape': (84, 84),


    'lr': 1e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'ppo_clip': 0.1,
    'w_ent': 0.01,    
    'epochs': 20,
    'n_iters': int(1e7),
    'batch_size': 512,

    'horizon': 256,
    'n_envs': 8,
    'groups': 1,

    'save_period': 20,  
    'eval_period': 20,    
    'log_period': 5,

    'a_std': [0.25, 0.15, 0.15], 

    'ckpt_save_path': "ckpt/checkpoint13-4.h5",
    # 'ckpt_load_path': "ckpt/checkpoint12-2.h5"
    'ckpt_load_path': "ckpt/eval-320.h5"
})





def preprocess_frame(img):     
    img = img[:-12, 6:-6] # (84, 84, 3)
    img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    img = img / 255.0
    img = img * 2 - 1    
    # print(f'img.shape: {img.shape}') 
    assert img.shape == (84, 84)
    return img






class Backbone(tf.keras.layers.Layer):

    def __init__(self):
        super(Backbone, self).__init__(name='backbone')

    def build(self, input_shape):

        self.seq = [] 
        self.seq.append(tf.keras.layers.Conv2D(filters=16, kernel_size=8, strides=4))
        self.seq.append(tf.keras.layers.LeakyReLU())
        self.seq.append(tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2))
        self.seq.append(tf.keras.layers.LeakyReLU())
        self.seq.append(tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2))
        self.seq.append(tf.keras.layers.ReLU())  
       
    def call(self, x, training=False): # x.shape: (64, 65, 64) = (64, 65, hidden_size)
        for layer in self.seq: x = layer(x, training=training)
        return x



 


class PoliycyNet(tf.keras.Model):

    def __init__(self):  

        super().__init__(name='policy_net')

        self.seq = []
        self.seq.append(Backbone())
        self.seq.append(tf.keras.layers.Flatten())
        self.seq.append(tf.keras.layers.Dense(128, activation='tanh', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0)))
        self.seq.append(tf.keras.layers.Dense(128, activation='tanh', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0)))
        self.seq.append(tf.keras.layers.Dense(num_actions, activation='tanh', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0)))

    def call(self, x, training=False): # x.shape: (64, 65, 64) = (64, 65, hidden_size)
        for layer in self.seq: x = layer(x, training=training)
        return x


class ValueNet(tf.keras.Model):

    def __init__(self):  

        super().__init__(name='value_net')

        self.seq = []
        self.seq.append(Backbone())
        self.seq.append(tf.keras.layers.Flatten())
        self.seq.append(tf.keras.layers.Dense(128, activation='tanh'))
        self.seq.append(tf.keras.layers.Dense(128, activation='tanh'))
        self.seq.append(tf.keras.layers.Dense(1))


    def call(self, x, training=False): # x.shape: (64, 65, 64) = (64, 65, hidden_size)
        for layer in self.seq: x = layer(x, training=training)
        return x


class Agent(tf.keras.Model):

    def __init__(self):  

        super().__init__()

        self.policy_net = PoliycyNet()

        self.a_std = self.add_weight("a_std", shape=[num_actions,], trainable=False,
                      initializer = tf.keras.initializers.Constant(value=0.5), dtype=tf.float32)
        
        self.value_net = ValueNet()

        self.update_counts = 0
        
        self.opt_val = tf.keras.optimizers.Adam(learning_rate=para.lr)
        self.opt_pol = tf.keras.optimizers.Adam(learning_rate=para.lr)

        self.i = 0
        self.prev_action = np.array([0.0, 0.0, 0.0])
        self.recent_frames = deque(maxlen=para.k) 
        for _ in range(para.k):
            self.recent_frames.append(np.zeros((para.img_shape)))


        self.load_checkpoint('111022533_hw3_data')


    def act(self, observation):

        observation = np.squeeze(observation)

        if(self.i % para.skip == 0):
            self.i = 1
            self.recent_frames.append(preprocess_frame(observation)) 
            s = np.stack(self.recent_frames, axis=-1)[np.newaxis,...]            
            a, _, _ = self.predict(s, greedy=False)            
            self.prev_action = a
  
        else:
            self.i += 1 
            
        return self.prev_action



    @tf.function
    def call(self, x, training=False):
        
        a_mean = self.policy_net(x, training=training)
        v = self.value_net(x, training=training)
    
        a_mean = a_min + ((a_mean + 1) / 2) * (a_max - a_min)
        return a_mean, tf.squeeze(v, axis=-1)



    def predict(self, state, greedy=False):
        
        state = tf.convert_to_tensor(state, tf.float32)                 
        a_mean, v = self(state)

        dist = tf.compat.v1.distributions.Normal(a_mean, self.a_std, validate_args=True)

        if(greedy):
            a = a_mean
        else:
            a = tf.squeeze(dist.sample(1)) # (b, num_actions)  

        a_logP = tf.reduce_sum(dist.log_prob(a), axis=-1) # (b,) 
                
        return a.numpy(), a_logP.numpy(), v.numpy()
 


    def train_step(self, batch):
        self.update_counts += 1
        pol_loss, val_loss = self._train_step(batch)
        return pol_loss.numpy(), val_loss.numpy()


    @tf.function
    def _train_step(self, batch):
        
      
        sta, a, a_logP_old, val_old, ret, adv = batch

        eps = para.ppo_clip


        with tf.GradientTape() as tape: 
            val = self.value_net(sta, training=True)
            val = tf.squeeze(val, axis=-1) 
            val_loss = tf.reduce_mean(tf.square(ret - val))  
  
        val_grads = tape.gradient(val_loss, self.value_net.trainable_variables)
        self.opt_val.apply_gradients(zip(val_grads, self.value_net.trainable_variables))

  
        with tf.GradientTape() as tape:
  
            a_mean = self.policy_net(sta, training=True)        
            a_mean = a_min + ((a_mean + 1) / 2) * (a_max - a_min)  

            dist = tf.compat.v1.distributions.Normal(a_mean, self.a_std, validate_args=True)
            a_logP = tf.reduce_sum(dist.log_prob(a), axis=-1) # (b,) 
            ent = dist.entropy()       
        
   
            r = tf.exp(a_logP - a_logP_old) # (b,)                        
            pol_loss = - tf.reduce_mean(tf.minimum(r * adv, tf.clip_by_value(r, 1-eps, 1+eps) * adv)) - \
                                        para.w_ent * tf.reduce_mean(tf.reduce_sum(ent, axis=-1))
 
            # pol_loss = - tf.reduce_mean(tf.minimum(r * adv, tf.clip_by_value(r, 1-eps, 1+eps) * adv))
  

        pol_grads = tape.gradient(pol_loss, self.policy_net.trainable_variables)
        self.opt_pol.apply_gradients(zip(pol_grads, self.policy_net.trainable_variables))


        return pol_loss, val_loss 


    def save_checkpoint(self, path):  
        print(f'- saved ckpt {path}') 
        self.save_weights(path)
         

    def load_checkpoint(self, path):         
        print(f'- loaded ckpt {path}') 
        self(tf.random.uniform(shape=[1, *para.img_shape, para.k]))
        self.load_weights(path)



 
