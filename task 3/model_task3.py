import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
import copy
import random
from collections import deque, namedtuple


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

# add nosie to the sample
class OUNoise: #TODO: may be can change it 
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = tf.convert_to_tensor(np.vstack([e.state for e in experiences if e is not None]),tf.double)
        actions = tf.convert_to_tensor(np.vstack([e.action for e in experiences if e is not None]),tf.double)
        rewards = tf.convert_to_tensor(np.vstack([e.reward for e in experiences if e is not None]),tf.double)
        next_states = tf.convert_to_tensor(np.vstack([e.next_state for e in experiences if e is not None]),tf.double)
        dones = tf.convert_to_tensor(np.vstack([e.done for e in experiences if e is not None]),tf.double)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

