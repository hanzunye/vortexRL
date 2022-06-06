###
# go back to specific height > 4.2m height, episode finish
# every step more soil get score -0.2
# Every step subtract score -1
# when finish check if get enough soil, get a score. (bucket-1000)/10
###

###
# go back to specific height > 4.2m height, episode finish
###
import random
import numpy as np

class reward_():

    def __init__(self):
        #self.reward_list = []
        self.reward = 0.0
        self.finish = False
        self.back = False
        self.M_old  = 0.0
        self.H_old  = 0.0
        self.T_old = 0.0

    def get_score_height(self,ob):
        m = 1200
        height = 4.2
        M = ob[0]
        H = ob[1]

        if M >= m:
            self.back = True
        if not self.back:
            if M > self.M_old:
                self.reward = -0.5
                self.M_old = M
            if M == self.M_old and M !=0:
                self.reward = -0.8
                self.M_old = M    
            if M < self.M_old and M!=0:           
                self.reward = -1.5
                self.M_old = M
            else:
                self.reward = -1.2
                self.M_old = M            

        if self.back:
            if H >= height:
                self.M_old = M
                self.finish = True
                if M>1000:
                    self.reward = 3.0#-0.1  
                if M>500:
                    self.reward = 2.0#-0.2
                else:
                    self.reward = -0.3#-0.3    
            if H < height:
                if H < self.H_old:
                    if M >= self.M_old:
                        self.reward = -1.2
                        self.M_old = M
                    if M < self.M_old:
                        self.reward = -1.5
                        self.M_old = M
                    self.H_old = H
                if H == self.H_old:
                    if M >= self.M_old:
                        self.reward = -0.8
                        self.M_old = M
                    if M < self.M_old:
                        self.reward = -1.1
                        self.M_old = M
                    self.H_old = H
                if H > self.H_old:
                    if M >= self.M_old:
                        self.reward = -0.5
                        self.M_old = M
                    if M < self.M_old:
                        self.reward = -0.8
                        self.M_old = M
                    self.H_old = H
                
        return self.reward , self.finish  

    def get_score(self,ob):
        Truck = ob[2]
        M = ob[0]
        if Truck >= 500 and M ==0:
            self.reward = Truck/50.
            self.finish = True

        if Truck > self.T_old:
            self.reward = - 0.1 - 0.5*random.random()
            self.M_old = M
            self.T_old = Truck
        # if Truck <= self.T_old and M<self.M_old:
        #     self.reward = -2.4
        #     self.M_old = M
        #     self.T_old = Truck
        else:
            self.reward = -2.0-0.5*random.random()
            self.M_old = M
            self.T_old = Truck
        return self.reward , self.finish
            


    def reset(self):

        self.back = False
        self.reward = 0.0
        self.finish = False
        self.M_old  = 0.0
        self.H_old  = 0.0
        self.T_old = 0.0



        ### if I choose 2000 as limit training not good , if i choose 1500 but depend on the last weight ob[0]/30 as test not good