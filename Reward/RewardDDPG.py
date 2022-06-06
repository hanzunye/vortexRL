###
# go back to specific height > 4.2m height, episode finish
###


class reward_():

    def __init__(self):
        #self.reward_list = []
        self.reward = 0.0
        self.finish = False
        self.back = False
        self.M_old  = 0.0
        self.H_old  = 0.0

    def get_score_height(self,ob):

        """
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
            #if M < self.M_old and M!=0:    
            else:       
                self.reward = -1.2
                self.M_old = M
            # else:
            #     self.reward = -1.2
            #     self.M_old = M            

        if self.back:
            if H >= height:
                self.finish = True
                if M>1000:
                    self.reward = -0.1  
                if M>500:
                    self.reward = -0.2
                else:
                    self.reward = -0.3    
            if H < height:
                if H < self.H_old:
                    self.reward = -1.2 # it;s better to set it to -1
                    self.M_old = M
                    # if M >= self.M_old:
                    #     self.reward = -1.2
                    #     self.M_old = M
                    # if M < self.M_old:
                    #     self.reward = -1.5
                    #     self.M_old = M
                    self.H_old = H
                if H == self.H_old:
                    if M >= self.M_old:
                        self.reward = -0.5
                        self.M_old = M
                    if M < self.M_old:
                        self.reward = -0.8
                        self.M_old = M
                    self.H_old = H
                if H > self.H_old:
                    if M >= self.M_old:
                        self.reward = -0.3
                        self.M_old = M
                    if M < self.M_old:
                        self.reward = -0.5
                        self.M_old = M
                    self.H_old = H"""
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
                        self.reward = (M-self.M_old)/600-1.5
                        self.M_old = M
                    self.H_old = H
                if H == self.H_old:
                    if M >= self.M_old:
                        self.reward = -0.8
                        self.M_old = M
                    if M < self.M_old:
                        self.reward = (M-self.M_old)/600-1.1
                        self.M_old = M
                    self.H_old = H
                if H > self.H_old:
                    if M >= self.M_old:
                        self.reward = -0.5
                        self.M_old = M
                    if M < self.M_old:
                        self.reward = (M-self.M_old)/600-0.8
                        self.M_old = M
                    self.H_old = H
                
        return self.reward , self.finish   



    def reset(self):

        self.back = False
        self.reward = 0.0
        self.finish = False
        self.M_old  = 0.0
        self.H_old  = 0.0
