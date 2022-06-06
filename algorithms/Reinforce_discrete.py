###
# Written by Yunze Han
# This code is based on  the paper 'Richard S. Sutton, David A. McAllester, Satinder P. Singh, and Yishay Mansour.
# Policy Gradient Methods for Reinforcement Learning with Function Approximation.'
# It uses tensorflow 2.0+ and fits for discrete action space
###
import tensorflow as tf
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd 

Lrate = 0.005
class Reinforce():
    ###
    # parameter: 
    # s_size :state size
    # h_size1,h_size2 : hidden layer
    # a_size : action size
    ###
    #def __init__(self, env, s_size=3, h_size1=32,h_size2=128, h_size3 = 256, a_size=125):
    def __init__(self, env, s_size=3, h_size1=16,h_size2=32, a_size=125):
        self.model = self.build_model(s_size, h_size1, h_size2, a_size)
        self.env = env
        self.size = s_size
        self.opt = tf.keras.optimizers.Adam(learning_rate=Lrate)



    def build_model(self,s_size, h_size1,h_size2, a_size):
        model = tf.keras.Sequential([
			Dense(h_size1, activation = 'relu', input_shape = (s_size,)),
			Dense(h_size2, activation = 'relu'),
            #Dense(h_size3, activation = 'relu'),
			Dense(a_size,activation = 'softmax')])

        return model

    def call(self,x):
            
        return self.model(x)

    def act(self,state):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        state = tf.reshape(state, (1,3), name=None)
        probs = self.call(state)
        m = tfp.distributions.Categorical(probs) 
        action = m.sample() # get the action number in the action map
        return int(action), m.log_prob(action) 


    def saveModel(self,path = 'reinforcemodel'):
        self.model.save_weights(path)
    
    def saveplot(self,scores,path = 'reinforceresult.png'):
        fig = plt.figure()
        fig.add_subplot(111)
        plt.plot(np.arange(1, len(scores)+1), scores)
        plt.ylabel('Rward')
        plt.xlabel('Episode #')
        plt.savefig(path)

    def run(path):
        #TODO
        pass

    def train_model(self,n_episodes=1000, max_t=50, gamma=0.8, render_every=1):
        env = self.env
        scores_deque = deque(maxlen=30)
        scores = []
        if render_every == 0:
            env.render(False) 
        for i_episode in range(1, n_episodes+1): 

            rewards = []
            log_probs = []
            done = False
            state = env.reset()
            state = state[0:3]
            with tf.GradientTape() as tape:
                for i in range(max_t):
                    
                    action_nr,log_prob = self.act(state)
                    next_state, reward, done, _ = env.step_discrete_Reinforce(action_nr)
                    rewards.append(reward)
                    log_probs.append(log_prob)
                    state = next_state
                    if done:
                        #rewards[-1] *=i
                        print('finish dig with {} step '.format(i))
                        break

                scores_deque.append(sum(rewards))
                scores.append(sum(rewards))
                sum_rewards = 0
                disR =[]
                new_rewards = [r+2.5 for r in rewards]
                new_rewards.reverse()
                for i in new_rewards:
                    sum_rewards = i + gamma*sum_rewards
                    disR.append(sum_rewards)
                disR.reverse()
                #loss = [-a*b for a,b in zip(disR,log_probs)]
                loss_list = [-a*b for a,b in zip(disR,log_probs)]
                loss = tf.math.reduce_sum(loss_list)   # check if we calculate minimal of each step or for all steps together.

            grads = tape.gradient(loss,self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
            if not done:
                print('didnot finish the task')

            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            if np.mean(scores_deque) > -1:
                self.saveModel()
                self.saveplot(scores)
                return scores
        self.saveModel()
        self.saveplot(scores)
        dataframe = pd.DataFrame(scores) 
        dataframe.to_csv("resultreinforce.csv")
        return scores
