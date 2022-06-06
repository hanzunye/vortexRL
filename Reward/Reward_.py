###
# go back to specific height > 4.2m height, episode finish
# every step more soil get score -0.2
# Every step subtract score -1
# when finish check if get enough soil, get a score. (bucket-1000)/10
###

import numpy as np

class reward_():

    def __init__(self):
        #self.reward_list = []
        self.reward = 0.0
        self.finish = False
        self.back = False
        self.last_time  = 0.0

    def get_score_height(self,ob):
        limit = 1500  #1500 old one
        height = 4.2
        if ob[0] > limit:
            self.back = True
        if self.back:
            if ob[1] > height:
                self.finish = True
                self.reward = ob[0]/500+0.4    #TODO: forget to change the mistake  +50  ob[1]/10 =0.4  TODO:baseline
                a = self.last_time-ob[0]
                self.reward = -1/(1 + np.exp(-a))
                self.last_time = ob[0]
            else:
                self.reward = -0.5
                self.last_time = ob[0]
        else:
            if ob[0] > self.last_time:
                self.reward = (ob[0]-self.last_time)/200 -1
                self.last_time = ob[0]
            else:
                self.reward = -3.0
                self.last_time = ob[0]
        return self.reward , self.finish         
    def get_score(self,ob):
        self.back = True
        limit = 1000
        if ob[0] > limit:
            self.finish = True
            self.reward = 4
        elif ob[0] > self.last_time:
            self.reward = (ob[0]-self.last_time)/200 -1
            self.last_time = ob[0]
        else:
            self.reward = -3.0
            self.last_time = ob[0]
        return self.reward , self.finish 

    def reset(self):

        self.back = False
        self.reward = 0.0
        self.finish = False



### if I choose 2000 as limit training not good , if i choose 1500 but depend on the last weight ob[0]/30 as test not good

# 整个过程分成三个部分，因此奖励函数，最好分成三个阶段，即挖掘，提升，放下，因此整个过程需要三次save frames
# 这样这个过程就会完美，因此应该在环境中设定切换的方式，并且分别存储模型
# 分段奖励函数的好处，避免代理人困惑，很多时候达不到第二阶段，甚至来回磨洋工，一旦接触过第二阶段，但又完成不了，目标不明确，
# 因此agent会在训练过程中奖励函数出现波动。