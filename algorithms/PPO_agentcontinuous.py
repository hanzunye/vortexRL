
import tensorflow_probability as tfp
import numpy as np
import tensorflow as tf
from tensorflow import keras
from Env_ import environment
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from collections import deque


class Buffer():
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.done = [] 

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.done[:]  


class Actor():

    def __init__(self, Critic,s_size=4, h_size1=64,h_size2=64, a_size=3,action_std_init = 0.6):
        self.last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        self.model = self.build_model(s_size, h_size1,h_size2, a_size)
        self.action_var = tf.fill((a_size,), action_std_init * action_std_init)
        self.a_size = a_size
        self.critic = Critic


    def build_model(self,s_size, h_size1,h_size2, a_size):
        model = tf.keras.Sequential([
			Dense(h_size1, activation = 'tanh', input_shape = (s_size,)),
			Dense(h_size2, activation = 'tanh'),
			Dense(a_size,activation = 'tanh',kernel_initializer=self.last_init)])
        return model

    def set_action_std(self,new_action_std):
        self.action_var = self.action_var = tf.fill((self.a_size,), new_action_std*new_action_std)
    
    def act(self,state):
        action_mean = self.call(state)
        cov_mat = tf.convert_to_tensor(np.sqrt(self.action_var))
        dist = tfp.distributions.Normal(action_mean,cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action_logprob.numpy() , action.numpy()
    
    def evaluate(self, state, action):
        action_mean = self.call(state)
        action_var = tf.broadcast_to(self.action_var,tf.shape(action_mean))
        dist = tfp.distributions.Normal(action_mean,action_var)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic.call(state)

        return action_logprobs, state_values, dist_entropy
        
    
    def call(self, x):
            
        return self.model(x)

class Critic():
    def __init__(self, s_size=4, h_size1=64,h_size2=64, a_size=3):
        self.model = self.build_model(s_size, h_size1,h_size2, a_size)


    def build_model(self,s_size, h_size1,h_size2, a_size):

        model = tf.keras.Sequential([
        Dense(h_size1, activation = 'tanh', input_shape = (s_size,)),
        Dense(h_size2, activation = 'tanh'),
        Dense(1)])
        return model

    def call(self, x):
            
        return self.model(x)


class PPO():
    def __init__(self, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.6):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = Buffer()

        critic = Critic()
        critic_old = Critic()
        critic_old.model.set_weights(critic.model.get_weights())
        self.actor = Actor(critic,action_std_init = action_std_init)
        self.actor_old = Actor(critic_old,action_std_init = action_std_init)
        self.actor_old.model.set_weights(self.actor.model.get_weights()) 
        
        self.actor_optimizer = tf.keras.optimizers.Adam(lr_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(lr_critic)

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.actor.set_action_std(new_action_std)
        self.actor_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
        else:
            self.set_action_std(self.action_std)

    def select_action(self,state):

        action_logprob, action = self.actor_old.act(state)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action
    
    def learn(self):
        rewards = []
        discounted_reward = 0
        # caculate the discount reward for all episodes
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.done)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        # normalization reward
        rewards = tf.convert_to_tensor(rewards,dtype = tf.float32)
        rewards = (rewards - tf.math.reduce_mean(rewards)) / (tf.math.reduce_std(rewards) + 1e-7)
        # change list to np array batch, (size episode*step,state_size)
        old_states = tf.convert_to_tensor(np.vstack(self.buffer.states),tf.float32)
        old_actions = tf.convert_to_tensor(np.vstack(self.buffer.actions),tf.float32)
        old_logprobs = tf.convert_to_tensor(np.vstack(self.buffer.logprobs),tf.float32)
        
        for _ in range(self.K_epochs):

            with tf.GradientTape() as g, tf.GradientTape() as f:

                logprobs, state_values, dist_entropy = self.actor.evaluate(old_states, old_actions)
                ratios = tf.exp(tf.squeeze(logprobs) - tf.squeeze(old_logprobs))
                advatages = rewards - tf.squeeze(state_values)
                advatages = tf.reshape(advatages,(-1,1))
                surr1 = ratios * advatages
                surr2 = tf.clip_by_value(ratios,1-self.eps_clip, 1+self.eps_clip) * advatages

                loss = -tf.math.minimum(surr1,surr2)  + 0.5* tf.keras.losses.mean_squared_error(tf.squeeze(state_values), rewards) - 0.01*tf.squeeze(dist_entropy)

            actor_grad = g.gradient(loss, self.actor.model.trainable_variables)
            critic_grad = f.gradient(loss, self.actor.critic.model.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad,self.actor.model.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(critic_grad,self.actor.critic.model.trainable_variables))

        self.actor_old.model.set_weights(self.actor.model.get_weights())
        self.actor_old.critic.model.set_weights(self.actor.critic.model.get_weights())

        self.buffer.clear()
    
    



env = environment()
agent = PPO(lr_actor=0.004, lr_critic=0.004, gamma=0.99, K_epochs=80, eps_clip=0.2, action_std_init=0.6)

def train(env,agent,render_every=0,n_episodes=1000,max_t = 100,print_every = 3):
    scores_deque = deque(maxlen=100)
    scores = []
    env.render(False)
    for i_episode in range(1,n_episodes+1) :
        state = env.reset()
        score = 0
        # if render_every == 0:
        #     env.render(False)
        # else:
        #     if (i_episode-1) % render_every == 0:
        #         env.render(True)
        #     else:
        #         env.render(False)
        for i in range(1, max_t+1):
            state = tf.convert_to_tensor(state, dtype=tf.float32)
            state = tf.reshape(state, (1,4), name=None)
            action = agent.select_action(state) # save buffer state, action, logprobs
            next_state, reward, done, _ = env.step_continuous(action)
            agent.buffer.rewards.append(reward)
            # a small trick: set the last step as True, make esaier to upate in the Buffer
            if i == max_t:
                agent.buffer.done.append(True)
            else:
                agent.buffer.done.append(done)
            state = next_state
            score += reward
            # if finish task and 4 episodes, then do agent update and break
            if done:
                if i_episode%4 == 0:
                    agent.learn()
                    break
            # if not finish but mat_t, 4 episodes, then update for 4 episodes
            elif i_episode%4 == 0 and i == max_t:
                agent.learn()
            
        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))

        #print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - print_every, np.mean(scores_deque)))
        
    return scores

        
scores = train(env, agent)

# load the weight and run the excavator
#agent.run(env,save_path = Save_path,max_t=100)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('ppo3.png')



