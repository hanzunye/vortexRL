

class rewardtest():

    def __init__(self):
        #self.reward_list = []
        self.reward = 0.0
        self.finish = False
        self.back = 1
        self.last_time  = 0.0
      
    def get_score(self,ob):
        limit = 1000
        if ob[0] > limit:
            self.finish = True
            self.reward = -0.1 # it was before setting as 1
        elif ob[0] > self.last_time:
            self.reward = -1.0
            self.last_time = ob[0]
        else:
            self.reward = -2.4
            self.last_time = ob[0]
        # elif ob[0] == 0:
        #     self.reward = -2
        #     self.last_time = ob[0]
        # elif ob[0] != 0 and ob[0]<self.last_time:
        #     self.reward = -1.5
        #     self.last_time = ob[0]
        # elif ob[0] != 0 and ob[0]==self.last_time:
        #     self.reward = -0.8
        #     self.last_time = ob[0]        
        # else:
        #     self.reward = -2
        #     self.last_time = ob[0]
        return self.reward , self.finish 

    def reset(self):

        self.reward = 0.0
        self.finish = False