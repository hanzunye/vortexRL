###
# This is based on the default env from vortex studio demo
# This is simple env for training
# action space can discrete or continually, beginning frame is near the soil
# new reward function
###

import Vortex
import vxatp3
import os
import numpy as np
from Reward_ import reward_
from itertools import product

# define discrete action_space
action_dict = []
for item in product([-0.5,0.5,0],repeat=3):  # 27 list
    a = list(item)
    # if a[2] ==-0.3:
    #     a[2] = -0.5
    # if a[2] == 0.3:
    #     a[2] = 0.5
    action_dict.append(a)


# This is the path of scene and setup file, change if needed
SCENE_PATH = '../Excavator Scene for vortex studi/Scenario/Excavator Scene/ExcavatorWorkshop - Copy.vxscene'
SETUP_FILE = '../Excavator Scene for vortex studio/excavator.vxc'
Sub_steps = 30  # each step how many frames

class environment():

    def __init__(self):

        # check the path valid
        self.setup_file = SETUP_FILE
        self.content_file = SCENE_PATH
        self.check_path(self.content_file,self.setup_file)
        self.application = vxatp3.VxATPConfig.createApplication(self, 'Excavator App', self.setup_file)
       
        # init some parameters
        self.done = False
        self.reward_ = reward_()
       
        # init display
        self.display = Vortex.VxExtensionFactory.create(Vortex.DisplayICD.kExtensionFactoryKey)
        self.display.getInput(Vortex.DisplayICD.kPlacementMode).setValue("Windowed")
        self.display.setName('3D Display')
        self.display.getInput(Vortex.DisplayICD.kPlacement).setValue(Vortex.VxVector4(50, 50, 1280, 7200))

        # load scene
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
    
    def reset(self):
        self.done = False
        # Switch to Simulation Mode
        vxatp3.VxATPUtils.requestApplicationModeChangeAndWait(self.application, Vortex.kModeSimulating)
        # Load first key frame
        self.keyFrameList.restore(self.key_frames_array[0])
        self.application.update()
        state = self.get_state_four()
        self.reward_.reset()
        return state

    def get_state(self):

        Swing_Position = self.interface.getOutputContainer()['Swing Position'].value
        Boom_position = self.interface.getOutputContainer()['Boom position'].value
        Stick_Position = self.interface.getOutputContainer()['Stick Position'].value
        Bucket_Position = self.interface.getOutputContainer()['Bucket Position'].value
        state = np.array([Swing_Position, Boom_position, Stick_Position, Bucket_Position])
        return state

    def get_state_four(self):
        Boom_position = self.interface.getOutputContainer()['Boom position'].value
        Stick_Position = self.interface.getOutputContainer()['Stick Position'].value
        Bucket_Position = self.interface.getOutputContainer()['Bucket Position'].value
        back = float(self.reward_.back)
        state = np.array([Boom_position, Stick_Position, Bucket_Position,back])
        return state
    
    # init the excavtor position near the soil
    def position_init(self,action):

        self.interface.getInputContainer()['Velocity Boom'].value = action[0]    # m/s Boom cylinder
        self.interface.getInputContainer()['Velocity Stick'].value = action[1]

        for _ in range(Sub_steps):
            self.application.update()   

    def step_discrete(self,action_nr):

        action = action_dict[action_nr]
        self.interface.getInputContainer()['Boom Signal'].value = action[0]    # m/s Boom cylinder
        self.interface.getInputContainer()['Stick Signal'].value = action[1]
        self.interface.getInputContainer()['Bucket Signal'].value = action[2]

        for _ in range(Sub_steps):
            self.application.update()   

        state = self.get_state_four()
        ob = self.get_observation()
        reward,self.done = self.reward_.get_score_height(ob)

        return state,reward,self.done,{} 
    def step_continuous(self,action):

        self.interface.getInputContainer()['Boom Signal'].value = float(action[0][0])  # m/s Boom cylinder
        self.interface.getInputContainer()['Stick Signal'].value = float(action[0][1])
        self.interface.getInputContainer()['Bucket Signal'].value = float(action[0][2])

        for _ in range(Sub_steps):
            self.application.update()   

        state = self.get_state_four()
        ob = self.get_observation()
        reward,self.done = self.reward_.get_score_height(ob)

        return state,reward,self.done,{} 
    def steptask3(self,action):

        self.interface.getInputContainer()['Boom Signal'].value = float(action[0][0])  # m/s Boom cylinder
        self.interface.getInputContainer()['Stick Signal'].value = float(action[0][1])
        self.interface.getInputContainer()['Bucket Signal'].value = float(action[0][2])
        self.interface.getInputContainer()['Swing Signal'].value = float(action[0][3])

        for _ in range(Sub_steps):
            self.application.update()   

        state = self.get_state()
        ob = self.get_observation()
        reward,self.done = self.reward_.get_score(ob)

        return state,reward,self.done,{} 
   
    def get_observation(self):
        bucket = self.interface.getOutputContainer()['Bucket'].value
        height = self.interface.getOutputContainer()['Bucket Height'].value
        truck = self.interface.getOutputContainer()['Truck'].value
        return np.array([bucket, height,truck])

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