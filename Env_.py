###
# Written by Yunze Han, used for the excavator scene in Vortex Studio
# This enviromnet is suitalbe for discrete and continuous action space
# Set the begining frame near the digging soil
# Works with the Reward function Reward_
###

import Vortex
import vxatp3
import os
import numpy as np
from Reward import Reward_
from Reward import RewardReinforce
from Reward import RewardDDPG
from itertools import product

# define discrete action_space, only used for algorithm PPO_discrete and Reinforce.
action_dict = []
#for item in product([-1,-0.5,1,0.5,0],repeat=3):
for item in product([-1.,1.,0],repeat=3):
    a = list(item)
    action_dict.append(a)
action_dict = np.asarray(action_dict)
# for i in range(3):
#     action_dict[:,i] = (i+1)*0.25*action_dict[:,i]
# action_dict[:,0] = 0.25 * action_dict[:,0]
# action_dict[:,1] = 0.4 * action_dict[:,1]
# action_dict[:,2] = 0.5 * action_dict[:,2]
# action_dict is a 125*3 numpy array, as action space


# The path of scene and setup file
SCENE_PATH = 'Excavator Scene for vortex studio/Scenario/Excavator Scene/ExcavatorWorkshop.vxscene'
# Caution: if your Vortex Studio isn't installed in default C:CM Labs, you need open your vxc file change python interpreter position in the properties
SETUP_FILE = 'Excavator Scene for vortex studio/excavator.vxc'



class environment():

    # default 1s with 60 frames, Sub_steps = 30 means half second for one sub_step.
    def __init__(self,Sub_steps = 30):

        # check the path valid
        self.setup_file = SETUP_FILE
        self.content_file = SCENE_PATH
        self.check_path(self.content_file,self.setup_file)
        self.application = vxatp3.VxATPConfig.createApplication(self, 'Excavator App', self.setup_file)
       
        # init Reward
        self.done = False
        self.reward_ = Reward_.reward_()
        self.rewardtest = RewardReinforce.rewardtest()
        self.rewardddpg = RewardDDPG.reward_()
       
        # init display
        self.display = Vortex.VxExtensionFactory.create(Vortex.DisplayICD.kExtensionFactoryKey)
        self.display.getInput(Vortex.DisplayICD.kPlacementMode).setValue("Windowed")
        self.display.setName('3D Display')
        self.display.getInput(Vortex.DisplayICD.kPlacement).setValue(Vortex.VxVector4(50, 50, 1280, 7200))

        # init Sub_step
        self.Sub_steps = Sub_steps

        # load scene, check whether loading correctly
        vxatp3.VxATPUtils.requestApplicationModeChangeAndWait(self.application, Vortex.kModeEditing)
        self.vxscene = self.application.getSimulationFileManager().loadObject(self.content_file)
        self.scene = Vortex.SceneInterface(self.vxscene)
        if not self.scene.valid():
            print("Loading scene failed :(")
        else:
            print("Scene loaded correctly.")
        
        # load RL interface
        self.interface = self.scene.findExtensionByName('RL Interface')  # -> Access custom interface
        print("RL INterface loaded correctly.")
        
        # switch to simulation mode
        vxatp3.VxATPUtils.requestApplicationModeChangeAndWait(self.application, Vortex.kModeSimulating)

        # Initialize first key frame
        self.position_init([0.12,1])
        self.keyFrameList = self.application.getContext().getKeyFrameManager().createKeyFrameList("KeyFrameList",False)
        self.application.update()
        self.keyFrameList.saveKeyFrame()
        self.waitForNbKeyFrames(1, self.application, self.keyFrameList)
        self.key_frames_array = self.keyFrameList.getKeyFrames()

    def check_path(self,path1,path2):
        if not os.path.isfile(path1):
            print("Error loading content file! :(")
        if not os.path.isfile(path2):
            print("Error loading setup file! :(")
    
    def waitForNbKeyFrames(self, expectedNbKeyFrames, application, keyFrameList):
        maxNbIter = 100
        nbIter = 0
        while len(keyFrameList.getKeyFrames()) != expectedNbKeyFrames and nbIter < maxNbIter:
            if not application.update():
                break
            ++nbIter

    # init the excavtor position near the soil
    def position_init(self,action):

        self.interface.getInputContainer()['Velocity Boom'].value = action[0]    # m/s Boom cylinder
        self.interface.getInputContainer()['Velocity Stick'].value = action[1]

        for _ in range(30):
            self.application.update() 

    # restore the save frame and return state
    def reset(self):
        self.done = False
        # Switch to Simulation Mode
        vxatp3.VxATPUtils.requestApplicationModeChangeAndWait(self.application, Vortex.kModeSimulating)
        # Load first key frame
        self.keyFrameList.restore(self.key_frames_array[0])
        self.application.update()
        state = self.get_state_four()
        self.reward_.reset()
        self.rewardtest.reset()
        self.rewardddpg.reset()
        return state

    # get state for RL interface, back is a signal means calculate from other inputs (if the bucket has enough soil itself)
    def get_state_four(self):
        Boom_position = self.interface.getOutputContainer()['Boom position'].value
        Stick_Position = self.interface.getOutputContainer()['Stick Position'].value
        Bucket_Position = self.interface.getOutputContainer()['Bucket Position'].value
        back = float(self.reward_.back)
        state = np.array([Boom_position, Stick_Position, Bucket_Position,back])
        return state
    
    # add a low-pass for our signal, so we didn't controll the actor
    def step_discrete(self,action_nr):

        action = action_dict[action_nr]
        self.interface.getInputContainer()['Boom Signal'].value = action[0]    # m/s Boom cylinder
        self.interface.getInputContainer()['Stick Signal'].value = action[1]
        self.interface.getInputContainer()['Bucket Signal'].value = action[2]

        for _ in range(self.Sub_steps):
            self.application.update()   

        state = self.get_state_four()
        ob = self.get_observation()
        reward,self.done = self.reward_.get_score(ob)

        return state,reward,self.done,{} 

    def get_state_three(self):
        Boom_position = self.interface.getOutputContainer()['Boom position'].value
        Stick_Position = self.interface.getOutputContainer()['Stick Position'].value
        Bucket_Position = self.interface.getOutputContainer()['Bucket Position'].value
        state = np.array([Boom_position, Stick_Position, Bucket_Position])
        return state
    
    # add a low-pass for our signal, so we didn't controll the actor
    def step_discrete_Reinforce(self,action_nr):

        action = action_dict[action_nr]
        self.interface.getInputContainer()['Boom Signal'].value = action[0]    # m/s Boom cylinder
        self.interface.getInputContainer()['Stick Signal'].value = action[1]
        self.interface.getInputContainer()['Bucket Signal'].value = action[2]

        for _ in range(30):#self.Sub_steps):
            self.application.update()   

        state = self.get_state_three()
        ob = self.get_observation()
        reward,self.done = self.rewardtest.get_score(ob)

        return state,reward,self.done,{} 

    def step_continuous_DDPG(self,action):

        self.interface.getInputContainer()['Boom Signal'].value = float(action[0][0])  # m/s Boom cylinder
        self.interface.getInputContainer()['Stick Signal'].value = float(action[0][1])
        self.interface.getInputContainer()['Bucket Signal'].value = float(action[0][2])

        for _ in range(self.Sub_steps):
            self.application.update()   

        state = self.get_state_four_DDPG()
        ob = self.get_observation()
        reward,self.done = self.rewardddpg.get_score_height(ob)

        return state,reward,self.done,{} 

    def get_state_four_DDPG(self):
        Boom_position = self.interface.getOutputContainer()['Boom position'].value
        Stick_Position = self.interface.getOutputContainer()['Stick Position'].value
        Bucket_Position = self.interface.getOutputContainer()['Bucket Position'].value
        back = float(self.rewardddpg.back)
        state = np.array([Boom_position, Stick_Position, Bucket_Position,back])
        return state

    def step_continuous(self,action):

        self.interface.getInputContainer()['Boom Signal'].value = float(action[0][0])  # m/s Boom cylinder
        self.interface.getInputContainer()['Stick Signal'].value = float(action[0][1])
        self.interface.getInputContainer()['Bucket Signal'].value = float(action[0][2])

        for _ in range(self.Sub_steps):
            self.application.update()   

        state = self.get_state_four()
        ob = self.get_observation()
        reward,self.done = self.reward_.get_score_height(ob)

        return state,reward,self.done,{} 
    
    def get_observation(self):
        bucket = self.interface.getOutputContainer()['Bucket'].value
        height = self.interface.getOutputContainer()['Bucket Height'].value
        return np.array([bucket, height])

    # default render is true, use False stop render.
    def render(self, active=True):
        # Find current list of displays
        current_displays = self.application.findExtensionsByName('3D Display')

        # If active, add a display and activate Vsync
        if active and len(current_displays) == 0:
            self.application.add(self.display)
            self.application.setSyncMode(Vortex.kSyncSoftwareAndVSync)

        # If not, remove the current display and deactivate Vsync
        elif not active:
            if len(current_displays) == 1:
                self.application.remove(current_displays[0])
            self.application.setSyncMode(Vortex.kSyncNone)

        # Find current list of displays
        current_displays = self.application.findExtensionsByName('3D Display')

        # If active, add a display and activate Vsync
        if active and len(current_displays) == 0:
            self.application.add(self.display)
            self.application.setSyncMode(Vortex.kSyncSoftwareAndVSync)

        # If not, remove the current display and deactivate Vsync
        elif not active:
            if len(current_displays) == 1:
                self.application.remove(current_displays[0])
            self.application.setSyncMode(Vortex.kSyncNone)