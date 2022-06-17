# from sdsofa.models import FloorModel, TargetModel, DroneModel, FingerModel, GripperModel, DroneGripperModel

import Sofa 
class SceneController(Sofa.PythonScriptController):
    def __init__(self, node, objects):
        super(SceneController, self).__init__()
        self.objects = objects

    def bwdInitGraph(self, node):
        for object in self.objects:
            print object
            object.model.create()


class BaseController(object):
    def __init__(self, controller_args):
        self.controller_args = controller_args

    def compute_control(self):
        pass

class FingerKeyboardController(BaseController):
    def __init__(self, *args, **kwargs):
        super(FingerKeyboardController, self).__init__()

        self.fingers = fingers
        self.name = "FingerController"

    def onKeyPressed(self, c):
        dir = None
        # UP key :
        if ord(c)==19:
            dir = [0.0,1.0,0.0]
        # DOWN key : rear
        elif ord(c)==21:
            dir = [0.0,-1.0,0.0]
        # LEFT key : left
        elif ord(c)==18:
            dir = [1.0,0.0,0.0]
        elif ord(c)==20:
            dir = [-1.0,0.0,0.0]


class FloorController(BaseController):
    def __init__(self, *args, **kwargs):
        super(Floor, self).__init__(*args, **kwargs)
        self.model = FloorModel(self.model_args)

class TargetController(BaseController):
    def __init__(self, *args, **kwargs):
        super(Target, self).__init__(*args, **kwargs)
        self.model = TargetModel(self.model_args)

class DroneController(BaseController):
    def __init__(self, *args, **kwargs):
        super(Drone, self).__init__(*args, **kwargs)
        self.model = DroneModel(self.model_args)

class FingerController(BaseController):
    def __init__(self, *args, **kwargs):
        super(Finger, self).__init__(*args, **kwargs)
        self.model = FingerModel(self.model_args)
        
class GripperController(BaseController):
    def __init__(self, *args, **kwargs):
        super(Gripper, self).__init__(*args, **kwargs)
        self.model = GripperModel(self.model_args)

class DroneGripperController(BaseController):
    def __init__(self, *args, **kwargs):
        super(DroneGripper, self).__init__(*args, **kwargs)
        self.model = DroneGripper(self.model_args)