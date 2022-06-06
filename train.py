###
# Written by Yunze Han, this code is for train different algorithms and save the result.
##

import algorithms.Reinforce_discrete
import algorithms.DDPG
import algorithms.PPO_agent
import algorithms.TRPO
import Env_

class Train_save:
    def __init__(self) -> None:
        self.env = Env_.environment()
        
    def train_Reinforce(self,s_size = 3,n_episodes=1000,max_t=50,render_every=0,a_size=125):
        env = self.env
        reinforce = algorithms.Reinforce_discrete.Reinforce(env,s_size=s_size,a_size=a_size)
        reinforce.train_model(n_episodes=n_episodes,max_t=max_t,render_every=render_every)

    def train_DDPG(self,n_episodes=1000, max_t=180,render_every=1):
        env = self.env
        DDPG = algorithms.DDPG.DDPG(env,random_seed = 15,state_size = 4, action_size = 3, n_agents=1)
        DDPG.train_ddpg(env,render_every=render_every,n_episodes=n_episodes, max_t=max_t)

    def train_PPO(self):
        env = self.env
        # here can change algorithm from PPO discrete to continuous
        PPO = algorithms.PPO_agent.PPO(epochs=1000,steps_per_epoch=50,clip_ratio=0.2)
        PPO.train_ppo(env)
    def train_TRPO(self):
        env = self.env
        TRPO = algorithms.TRPO.Train_TRPO(env)
        TRPO.train_trpo(env, num_episodes = 1000, gamma = 0.095, lam = 0.98, kl_targ = 0.003, batch_size =20, hid1_mult = 10 , init_logvar = -1.0 ,max_t=50, render_every= 0)

if __name__ == '__main__':
    Train = Train_save()
    '''choose from the four algorithms'''

    Train.train_Reinforce(s_size =3,n_episodes=1000, max_t=20,render_every=1,a_size=27) #reward +2.5， sub = 30 maxt =20

    #Train.train_DDPG(n_episodes=1000, max_t=80,render_every=1) # old reward seed 25

    #Train.train_PPO()

    #Train.train_TRPO()
