###
# Written by Yunze Han
# This code is based on  the paper 'Continuous control with deep reinforcement learning
# Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra'
# It uses tensorflow 2.0+ and fits for continuous action space
###

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
import copy
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import random
from collections import deque
import pandas as pd

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.9            # discount factor
TAU = 5e-3              # for soft update of target parameters
LR_ACTOR = 5e-4         # learning rate of the actor
LR_CRITIC = 5e-4        # learning rate of the critic
NOISE_SD = 0.10         # noise scale
UPDATE_EVERY = 1

class Actor():

    def __init__(self, s_size=3, h_size1=400,h_size2=300, a_size=3,init_w=3e-3):
        self.last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        self.model = self.build_model(s_size, h_size1,h_size2, a_size)

    def build_model(self,s_size, h_size1,h_size2, a_size):
        model = tf.keras.Sequential([
			Dense(h_size1, activation = 'relu', input_shape = (s_size,)),
			Dense(h_size2, activation = 'relu'),
			Dense(a_size,activation = 'tanh',kernel_initializer=self.last_init)])
        return model

    def call(self, x):
            
        return self.model(x)

class Critic():
    def __init__(self, s_size=3, h_size1=400,h_size2=300, a_size=3):
        self.model = self.build_model(s_size, h_size1,h_size2, a_size)

    def build_model(self,s_size, h_size1,h_size2, a_size):
        state_input = layers.Input(shape=(s_size))
        state_out = layers.Dense(h_size1, activation="relu")(state_input)

        # Action as input
        action_input = layers.Input(shape=(a_size))

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_input])

        out = layers.Dense(h_size2, activation="relu")(concat)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    def call(self, x):
            
        return self.model(x)


class OUNoise():
    # add nosie to the sample

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        #Initialize parameters and noise process.
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        #Reset the internal state (= noise) to mean (mu).
        self.state = copy.copy(self.mu)
        
    def sample(self):
        #Update internal state and return it as a noise sample.
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

class ReplayBuffer():
    #Fixed-size buffer to store experience tuples.

    def __init__(self, action_size, buffer_size, batch_size,seed):
        #Initialize a ReplayBuffer object.
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        #Add a new experience to memory.
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        #Randomly sample a batch of experiences from memory.
        experiences = random.sample(self.memory, k=self.batch_size)

        states = tf.convert_to_tensor(np.vstack([e.state for e in experiences if e is not None]),tf.double)
        actions = tf.convert_to_tensor(np.vstack([e.action for e in experiences if e is not None]),tf.double)
        rewards = tf.convert_to_tensor(np.vstack([e.reward for e in experiences if e is not None]),tf.double)
        next_states = tf.convert_to_tensor(np.vstack([e.next_state for e in experiences if e is not None]),tf.double)
        dones = tf.convert_to_tensor(np.vstack([e.done for e in experiences if e is not None]),tf.double)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        #Return the current size of internal memory.
        return len(self.memory)

class DDPG():
    #Interacts with and learns from the environment.
    
    def __init__(self,env,random_seed, state_size = 4, action_size = 3, n_agents=1):
        
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.t_step = 0
        self.seed = random.seed(random_seed)


        # Actor Network (w/ Target Network)
        self.actor_local = Actor(s_size=state_size)
        self.actor_target = Actor(s_size=state_size)
        self.actor_optimizer = tf.keras.optimizers.Adam(LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(s_size=state_size)
        self.critic_target = Critic(s_size=state_size)
        self.critic_optimizer = tf.keras.optimizers.Adam(LR_CRITIC)

        # Noise process
        self.noise = OUNoise((n_agents, action_size),random_seed, sigma = NOISE_SD)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done):
        #Save experience in replay memory, and use random sample from buffer to learn.
        # Save experience / reward
        
        #for i in range(len(states)):
        #    self.memory.add(states[i, :], actions[i, :], rewards[i], next_states[i, :], dones[i])

        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if len(self.memory) > BATCH_SIZE and self.t_step == 0:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        #Returns actions for given state as per current policy.
        #state = tf.convert_to_tensor(state,tf.float)
        action = self.actor_local.call(state)
        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):


        state, action, reward, next_state, done = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with tf.GradientTape() as tape:
            target_action = self.actor_target.call(next_state)
            a = self.critic_target.call([next_state, target_action])
            a = tf.cast(a,dtype=tf.float64)
            y = reward + gamma * a
            critic_value = self.critic_local.call([state, action])
            critic_value = tf.cast(critic_value, dtype= tf.float64)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_local.model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_local.model.trainable_variables))

        # ---------------------------- update actor ---------------------------- #
        with tf.GradientTape() as tape:
            action = self.actor_local.call(state)
            critic_value = self.critic_local.call([state, action])
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_local.model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_local.model.trainable_variables))

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local.model, self.critic_target.model, TAU)
        self.soft_update(self.actor_local.model, self.actor_target.model, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        #Soft update model parameters.
        #θ_target = τ*θ_local + (1 - τ)*θ_target


        for target_param, local_param in zip(target_model.trainable_variables, local_model.trainable_variables):

            #target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            target_param.assign(local_param * tau + target_param * (1 - tau))
   
    def run(self,env,save_path = 'ddpgmodel',max_t=1000):
        self.actor_local.model.load_weights(save_path)
        state = env.reset()
        score = 0
        for i in range(max_t):
            state = tf.convert_to_tensor(state, dtype=tf.float32)
            state = tf.reshape(state, (1,4), name=None)
            action = self.act(state)
            next_state, reward, done, _ = env.step_continuous_DDPG(action)
            state = next_state
            score += reward
            if done:
                print('!')
                print('!')
                print('finish the task in ',i,' step with socre',score)
                break     

    def saveModel(self,path = 'ddpgmodel'):
        self.actor_local.model.save_weights(path)
    
    def saveplot(self,scores,path = 'ddpgresult.png'):
        fig = plt.figure()
        fig.add_subplot(111)
        plt.plot(np.arange(1, len(scores)+1), scores)
        plt.ylabel('Reward')
        plt.xlabel('Episode #')
        plt.savefig(path)    
               
    def train_ddpg(self,env, render_every=10, n_episodes=1000, max_t=180, print_every=1,save_path='ddpgmodel',result = 'ddpgresult.png'):
        scores_deque = deque(maxlen=100)
        scores = []
        if render_every == 0:
            env.render(False)
        for i_episode in range(1, n_episodes+1): 
            rewards = []
            state = env.reset()

            for i in range(max_t):
                state = tf.convert_to_tensor(state, dtype=tf.float32)
                state = tf.reshape(state, (1,4), name=None)
                action = self.act(state)
                next_state, reward, done, _ = env.step_continuous_DDPG(action)

                self.step(state, action, reward, next_state, done)
                state = next_state
                rewards.append(reward)
                if done:
                    print('finish in {} step'.format(i))
                    break
            scores_deque.append(np.sum(rewards))
            scores.append(np.sum(rewards))
            
            print('\rEpisode {}\tAverage Score: {:.2f} score {}'.format(i_episode, np.mean(scores_deque),np.sum(rewards)))
            # if np.mean(scores_deque) >= -10:
            #     self.saveModel(save_path)
            #     self.saveplot(scores,result)
            #     return scores
        self.saveModel(save_path)
        self.saveplot(scores,result)
        dataframe = pd.DataFrame(scores) 
        dataframe.to_csv("resultddpg.csv")
                
        return scores