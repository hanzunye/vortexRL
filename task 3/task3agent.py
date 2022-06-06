import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from Env_task3 import environment
from collections import deque
from model_task3 import Actor, Critic, OUNoise,ReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.9            # discount factor
TAU = 5e-3              # for soft update of target parameters
LR_ACTOR = 5e-4         # learning rate of the actor
LR_CRITIC = 5e-4        # learning rate of the critic
NOISE_SD = 0.10         # noise scale
UPDATE_EVERY = 1
class task3():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size =4, action_size=4, n_agents=1, random_seed=25):

        self.state_size = state_size
        self.action_size = action_size
        self.t_step = 0
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(s_size=state_size,a_size=action_size)
        self.actor_target = Actor(s_size=state_size,a_size=action_size)
        self.actor_optimizer = tf.keras.optimizers.Adam(LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(s_size=state_size,a_size=action_size)
        self.critic_target = Critic(s_size=state_size,a_size=action_size)
        self.critic_optimizer = tf.keras.optimizers.Adam(LR_CRITIC)

        # Noise process
        self.noise = OUNoise((n_agents, action_size),random_seed , sigma = NOISE_SD)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
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
        """Returns actions for given state as per current policy."""
        #state = tf.convert_to_tensor(state,tf.float)
        action = self.actor_local.call(state)
        if add_noise:
            action += self.noise.sample()

   
        return np.clip(action,-1,1)

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
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.trainable_variables, local_model.trainable_variables):

            #target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            target_param.assign(local_param * tau + target_param * (1 - tau))
   
    def run(self,env,save_path = 'actor_local_weights',max_t=1000):
        self.actor_local.model.load_weights(save_path)
        state = env.reset()
        score = 0
        for i in range(max_t):
            state = tf.convert_to_tensor(state, dtype=tf.float32)
            state = tf.reshape(state, (1,4), name=None)
            action = self.act(state)
            next_state, reward, done, _ = env.step_continuous(action)
            state = next_state
            score += reward
            if done:
                break

env = environment()
agent = task3(4,4,1,18)
def run(env,path ='easyeasyddpgmodelx'):
    actor_local = Actor(s_size=4,a_size=3)
    actor_local.model.load_weights(path)
    state = env.reset()
    for i in range(100):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        state = tf.reshape(state, (1,4), name=None)
        action = actor_local.call(state)
        next_state, _, done, _ = env.step_continuous(action)
        state = next_state
        if done:
            break
    return env.reward_.M_old

def train_ddpg(env, agent, render_every=0, n_episodes=10000, max_t=1000, print_every=100,save_path='actor_weights',result = 'mygraph.png'):
    scores_deque = deque(maxlen=30)
    scores = []
    if render_every == 0:
        env.render(False)
    for i_episode in range(1, n_episodes+1): 
        score = 0
        M = run(env)
        if M < 1000:
            continue
        action = np.array([[0,0,0,0.9]])
        env.steptask3(action)
        state,_,_,_ = env.steptask3(action)
        env.reward_.reset()
        for i in range(max_t):
            state = tf.convert_to_tensor(state, dtype=tf.float32)
            state = tf.reshape(state, (1,4), name=None)
            action = agent.act(state)
            action[0][0] *= 0.01
            action[0][1] *= 0.2
            action[0][3] *= 0.01
            next_state, reward, done, _ = env.steptask3(action)

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                print(i,'steps')
                break
        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, scores[-1]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('task3final.png')
    agent.actor_local.model.save_weights(save_path)        
    return scores

scores = train_ddpg(env, agent,render_every=0, n_episodes=1000,print_every=1,max_t=50)