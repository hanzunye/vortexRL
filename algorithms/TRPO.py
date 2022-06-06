import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import tensorflow as tf
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from datetime import datetime
import tensorflow as tf
from Env_ import environment
from collections import deque

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

class Logger(object):
    def __init__(self, logname, now):

        path = os.path.join('log-files', logname, now)
        os.makedirs(path)
        path = os.path.join(path, 'log.csv')
        self.write_header = True
        self.log_entry = {}
        self.f = open(path, 'w')
        self.writer = None  

    def write(self, display=True):

        if display:
            self.disp(self.log_entry)
        if self.write_header:
            fieldnames = [x for x in self.log_entry.keys()]
            self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
            self.writer.writeheader()
            self.write_header = False
        self.writer.writerow(self.log_entry)
        self.log_entry = {}

    @staticmethod
    def disp(log):
        #Print metrics to stdout
        log_keys = [k for k in log.keys()]
        log_keys.sort()
        print('***** Episode {}, Mean R = {:.1f} *****'.format(log['_Episode'],
                                                               log['_MeanReward']))
        for key in log_keys:
            if key[0] != '_':  # don't display log items with leading
                print('{:s}: {:.3g}'.format(key, log[key]))
        print('\n')

    def log(self, items):
        
        # Update fields in log (does not write to file, used to collect updates.


        self.log_entry.update(items)

    def close(self):
        # Close log file - log cannot be written after this
        self.f.close()

class Scaler(object):
    #Generate scale and offset based on running mean and stddev along axis=0


    def __init__(self, obs_dim):

        self.vars = np.zeros(obs_dim)
        self.means = np.zeros(obs_dim)
        self.m = 0
        self.n = 0
        self.first_pass = True

    def update(self, x):

        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))
            self.vars = np.maximum(0.0, self.vars)  # occasionally goes negative, clip
            self.means = new_means
            self.m += n

    def get(self):
        # returns 2-tuple: (scale, offset)
        return 1/(np.sqrt(self.vars) + 0.1)/3, self.means

class NNValueFunction(object):
    # NN-based state-value function
    def __init__(self, obs_dim, hid1_mult):

        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.obs_dim = obs_dim
        self.hid1_mult = hid1_mult
        self.epochs = 10
        self.lr = None  # learning rate set in _build_model()
        self.model = self._build_model()

    def _build_model(self):
        
        obs = Input(shape=(self.obs_dim,), dtype='float32')
        # hid1 layer size is 10x obs_dim, hid3 size is 10, and hid2 is geometric mean
        hid1_units = self.obs_dim * self.hid1_mult
        hid3_units = 5  # 5 chosen empirically on 'Hopper-v1'
        hid2_units = int(np.sqrt(hid1_units * hid3_units))
        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        self.lr = 1e-2 / np.sqrt(hid2_units)  # 1e-2 empirically determined
        print('Value Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}'
              .format(hid1_units, hid2_units, hid3_units, self.lr))
        y = Dense(hid1_units, activation='tanh')(obs)
        y = Dense(hid2_units, activation='tanh')(y)
        y = Dense(hid3_units, activation='tanh')(y)
        y = Dense(1)(y)
        model = Model(inputs=obs, outputs=y)
        optimizer = Adam(self.lr)
        model.compile(optimizer=optimizer, loss='mse')

        return model

    def fit(self, x, y, logger):
        # Fit model to current data batch + previous data batch

        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches
        y_hat = self.model.predict(x)  # check explained variance prior to update
        old_exp_var = 1 - np.var(y - y_hat)/np.var(y)
        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])
        self.replay_buffer_x = x
        self.replay_buffer_y = y
        self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=batch_size,
                       shuffle=True, verbose=0)
        y_hat = self.model.predict(x)
        loss = np.mean(np.square(y_hat - y))         # explained variance after update
        exp_var = 1 - np.var(y - y_hat) / np.var(y)  # diagnose over-fitting of val func

        logger.log({'ValFuncLoss': loss,
                    'ExplainedVarNew': exp_var,
                    'ExplainedVarOld': old_exp_var})

    def predict(self, x):
        # Predict method 
        return self.model.predict(x)

class Policy(object):
    def __init__(self, obs_dim, act_dim, kl_targ, hid1_mult, init_logvar):
 
        self.beta = 1.0  # dynamically adjusted D_KL loss multiplier
        eta = 50  # multiplier for D_KL-kl_targ hinge-squared loss
        self.kl_targ = kl_targ
        self.epochs = 20
        self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
        self.trpo = TRPO(obs_dim, act_dim, hid1_mult, kl_targ, init_logvar, eta)
        self.policy = self.trpo.get_layer('policy_nn')
        self.lr = self.policy.get_lr()  # lr calculated based on size of PolicyNN
        self.trpo.compile(optimizer=Adam(self.lr * self.lr_multiplier))
        self.logprob_calc = LogProb()

    def sample(self, obs):
        # Draw sample from policy.
        act_means, act_logvars = self.policy(obs)
        act_stddevs = np.exp(act_logvars / 2)

        return np.random.normal(act_means, act_stddevs).astype(np.float32)

    def update(self, observes, actions, advantages, logger):
        # Update policy based on observations, actions and advantages

        K.set_value(self.trpo.optimizer.lr, self.lr * self.lr_multiplier)
        K.set_value(self.trpo.beta, self.beta)
        old_means, old_logvars = self.policy(observes)  
        #print(old_means,old_logvars) 
        old_means = old_means.numpy()
        old_logvars = old_logvars.numpy()
        if old_logvars.shape[0]==1:
            old_logvars = np.tile(old_logvars, (old_means.shape[0],1))
        old_logp = self.logprob_calc([actions, old_means, old_logvars])
        old_logp = old_logp.numpy()
        loss, kl, entropy = 0, 0, 0
        for e in range(self.epochs):
            loss = self.trpo.train_on_batch([observes, actions, advantages,old_means, old_logvars, old_logp])
            kl, entropy = self.trpo.predict_on_batch([observes, actions, advantages,
                                                      old_means, old_logvars, old_logp])
            kl, entropy = np.mean(kl), np.mean(entropy)
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
       
        if kl > self.kl_targ * 2:  # servo beta to reach D_KL target
            self.beta = np.minimum(35, 1.5 * self.beta)  # max clip beta
            if self.beta > 30 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2:
            self.beta = np.maximum(1 / 35, self.beta / 1.5)  # min clip beta
            if self.beta < (1 / 30) and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5

        logger.log({'PolicyLoss': loss,
                    'PolicyEntropy': entropy,
                    'KL': kl,
                    'Beta': self.beta,
                    '_lr_multiplier': self.lr_multiplier})

class PolicyNN(Layer):
    # Neural net for policy approximation function.

    def __init__(self, obs_dim, act_dim, hid1_mult, init_logvar, **kwargs):
        super(PolicyNN, self).__init__(**kwargs)
        self.batch_sz = None
        self.init_logvar = init_logvar
        hid1_units = obs_dim * hid1_mult
        hid3_units = act_dim * 10  # 10 empirically determined
        hid2_units = int(np.sqrt(hid1_units * hid3_units))
        self.lr = 9e-4 / np.sqrt(hid2_units)  # 9e-4 empirically determined
        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        self.dense1 = Dense(hid1_units, activation='tanh', input_shape=(obs_dim,))
        self.dense2 = Dense(hid2_units, activation='tanh', input_shape=(hid1_units,))
        self.dense3 = Dense(hid3_units, activation='tanh', input_shape=(hid2_units,))
        self.dense4 = Dense(act_dim, input_shape=(hid3_units,))
        # logvar_speed increases learning rate for log-variances.
        # heuristic sets logvar_speed based on network size.
        logvar_speed = (10 * hid3_units) // 48
        self.logvars = self.add_weight(shape=(logvar_speed, act_dim),
                                       trainable=True, initializer='zeros')
        print('Policy Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}, logvar_speed: {}'
              .format(hid1_units, hid2_units, hid3_units, self.lr, logvar_speed))

    def build(self, input_shape):
        self.batch_sz = input_shape[0]

    def call(self, inputs, **kwargs):
        y = self.dense1(inputs)
        y = self.dense2(y)
        y = self.dense3(y)
        means = self.dense4(y)
        logvars = K.sum(self.logvars, axis=0, keepdims=True) + self.init_logvar
        logvars = K.tile(logvars, (self.batch_sz, 1))
        return [means, logvars]

    def get_lr(self):
        return self.lr


class KLEntropy(Layer):
    """
    Layer calculates:
        1. KL divergence between old and new distributions
        2. Entropy of present policy

    """
    def __init__(self, **kwargs):
        super(KLEntropy, self).__init__(**kwargs)
        self.act_dim = None

    def build(self, input_shape):
        self.act_dim = input_shape[0][1]

    def call(self, inputs, **kwargs):
        old_means, old_logvars, new_means, new_logvars = inputs
        log_det_cov_old = K.sum(old_logvars, axis=-1, keepdims=True)
        log_det_cov_new = K.sum(new_logvars, axis=-1, keepdims=True)
        trace_old_new = K.sum(K.exp(old_logvars - new_logvars), axis=-1, keepdims=True)
        kl = 0.5 * (log_det_cov_new - log_det_cov_old + trace_old_new +
                    K.sum(K.square(new_means - old_means) /
                          K.exp(new_logvars), axis=-1, keepdims=True) -
                    np.float32(self.act_dim))
        entropy = 0.5 * (np.float32(self.act_dim) * (np.log(2 * np.pi) + 1.0) +
                         K.sum(new_logvars, axis=-1, keepdims=True))

        return [kl, entropy]


class LogProb(Layer):
    # Layer calculates log probabilities of a batch of actions.
    def __init__(self, **kwargs):
        super(LogProb, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        actions, act_means, act_logvars = inputs
        logp = -0.5 * K.sum(act_logvars, axis=-1, keepdims=True)
        logp += -0.5 * K.sum(K.square(actions - act_means) / K.exp(act_logvars),
                             axis=-1, keepdims=True)

        return logp


class TRPO(Model):
    def __init__(self, obs_dim, act_dim, hid1_mult, kl_targ, init_logvar, eta, **kwargs):
        super(TRPO, self).__init__(**kwargs)
        self.kl_targ = kl_targ
        self.eta = eta
        self.beta = self.add_weight('beta', initializer='zeros', trainable=False)
        self.policy = PolicyNN(obs_dim, act_dim, hid1_mult, init_logvar)
        self.logprob = LogProb()
        self.kl_entropy = KLEntropy()

    def call(self, inputs):
        obs, act, adv, old_means, old_logvars, old_logp = inputs
        new_means, new_logvars = self.policy(obs)
        new_logp = self.logprob([act, new_means, new_logvars])
        kl, entropy = self.kl_entropy([old_means, old_logvars,
                                       new_means, new_logvars])
        loss1 = -K.mean(adv * K.exp(new_logp - old_logp))
        loss2 = K.mean(self.beta * kl)
        # TODO - Take mean before or after hinge loss?
        loss3 = self.eta * K.square(K.maximum(0.0, K.mean(kl) - 2.0 * self.kl_targ))
        self.add_loss(loss1 + loss2 + loss3)

        return [kl, entropy]

class Train_TRPO():
    def __init__(self,env):
        pass

    def run_policy(self,env, policy, scaler, logger, episodes,max_t=10,render_every = 1):
        #Run policy and collect data for a minimum of min_steps and min_episodes

        total_steps = 0
        trajectories = []
        for i_episode in range(1, episodes+1):
            obs = env.reset()
            observes, actions, rewards, unscaled_obs = [], [], [], []
            done = False
            step = 0.0
            scale, offset = scaler.get()
            scale[-1] = 1.0  # don't scale time step feature
            offset[-1] = 0.0  # don't offset time step feature
            if render_every == 0:
                env.render(False)
            else:
                if (i_episode-1) % render_every == 0:
                    env.render(True)
                else:
                    env.render(False)
            for _ in range(max_t):
                obs = np.concatenate([obs, [step]])  # add time step feature
                obs = obs.astype(np.float32).reshape((1, -1))
                unscaled_obs.append(obs)
                obs = np.float32((obs - offset) * scale)  # center and scale observations
                observes.append(obs)
                action = policy.sample(obs)
                actions.append(action)
                obs, reward, done, _ = env.step_continuous(action)
                rewards.append(reward)
                step += 1e-3  # increment time step feature
                if done:
                    break
            observes, actions, rewards, unscaled_obs = (np.concatenate(observes), np.concatenate(actions),np.array(rewards, dtype=np.float32), np.concatenate(unscaled_obs))

            total_steps += observes.shape[0]
            trajectory = {'observes': observes,
                        'actions': actions,
                        'rewards': rewards,
                        'unscaled_obs': unscaled_obs}
            #print('episode ', i_episode ,'get score', np.sum(trajectory['rewards']))
            trajectories.append(trajectory)
        unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
        scaler.update(unscaled)  # update running statistics for scaling observations
        logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                    'Steps': total_steps})

        return trajectories

    def discount(self,x, gamma):
        # Calculate discounted forward sum of a sequence at each point
        return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


    def add_disc_sum_rew(self, trajectories, gamma):
        # Adds discounted sum of rewards to all time steps of all trajectories

        for trajectory in trajectories:
            if gamma < 0.999:  # don't scale for gamma ~= 1
                rewards = trajectory['rewards'] * (1 - gamma)
            else:
                rewards = trajectory['rewards']
            disc_sum_rew = self.discount(rewards, gamma)
            trajectory['disc_sum_rew'] = disc_sum_rew

    def add_value(self,trajectories, val_func):
        # Adds estimated value to all time steps of all trajectories

        for trajectory in trajectories:
            observes = trajectory['observes']
            values = val_func.predict(observes)
            trajectory['values'] = values.flatten()

    def add_gae(self, trajectories, gamma, lam):
        # Add generalized advantage estimator.
        # https://arxiv.org/pdf/1506.02438.pdf

        for trajectory in trajectories:
            if gamma < 0.999:  # don't scale for gamma ~= 1
                rewards = trajectory['rewards'] * (1 - gamma)
            else:
                rewards = trajectory['rewards']
            values = trajectory['values']
            # temporal differences
            tds = rewards - values + np.append(values[1:] * gamma, 0)
            advantages = self.discount(tds, gamma * lam)
            trajectory['advantages'] = advantages

    def build_train_set(self, trajectories):

        observes = np.concatenate([t['observes'] for t in trajectories])
        actions = np.concatenate([t['actions'] for t in trajectories])
        disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
        advantages = np.concatenate([t['advantages'] for t in trajectories])
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        return observes, actions, advantages, disc_sum_rew

    def log_batch_stats(self, observes, actions, advantages, disc_sum_rew, logger, episode):
    # Log various batch statistics 
        logger.log({'_mean_obs': np.mean(observes),
                    '_min_obs': np.min(observes),
                    '_max_obs': np.max(observes),
                    '_std_obs': np.mean(np.var(observes, axis=0)),
                    '_mean_act': np.mean(actions),
                    '_min_act': np.min(actions),
                    '_max_act': np.max(actions),
                    '_std_act': np.mean(np.var(actions, axis=0)),
                    '_mean_adv': np.mean(advantages),
                    '_min_adv': np.min(advantages),
                    '_max_adv': np.max(advantages),
                    '_std_adv': np.var(advantages),
                    '_mean_discrew': np.mean(disc_sum_rew),
                    '_min_discrew': np.min(disc_sum_rew),
                    '_max_discrew': np.max(disc_sum_rew),
                    '_std_discrew': np.var(disc_sum_rew),
                    '_Episode': episode
                    })




    def train_trpo(self, env, num_episodes, gamma, lam, kl_targ, batch_size, hid1_mult, init_logvar, max_t=100,render_every = 1):
        # Main training loop

        obs_dim = 4
        act_dim = 3
        now = datetime.utcnow().strftime("%d%b_%H-%M-%S")  # create unique directories
        logger = Logger(logname= 'excavator', now=now)
        obs_dim += 1
        scaler = Scaler(obs_dim)
        val_func = NNValueFunction(obs_dim, hid1_mult)
        policy = Policy(obs_dim, act_dim, kl_targ, hid1_mult, init_logvar)
        # run a few episodes of untrained policy to initialize scaler:
        self.run_policy(env, policy, scaler, logger, episodes=1)
        
        episode = 0
        while episode < num_episodes:
            trajectories = self.run_policy(env, policy, scaler, logger, episodes=batch_size, max_t= max_t,render_every = render_every)
            episode += len(trajectories)
            self.add_value(trajectories, val_func)  # add estimated values to episodes
            self.add_disc_sum_rew(trajectories, gamma)  # calculated discounted sum of Rs
            self.add_gae(trajectories, gamma, lam)  # calculate advantage
            # concatenate all episodes into single NumPy arrays
            observes, actions, advantages, disc_sum_rew = self.build_train_set(trajectories)
            # add various stats to training log:
            self.log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode)
            policy.update(observes, actions, advantages, logger)  # update policy
            val_func.fit(observes, disc_sum_rew, logger)  # update value function
            logger.write(display=True)
        logger.close()
