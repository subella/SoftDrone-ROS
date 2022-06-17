from sdsofa.models import FloorModel, TargetModel, DroneModel, FingerModel, GripperModel, DroneGripperModel

class Base(object):
    def __init__(self,
                 model_args,
                 controller_args):

        self.model = None
        self.controller = None
        self.model_args = model_args
        self.controller_args = controller_args

class Floor(Base):
    def __init__(self, *args, **kwargs):
        super(Floor, self).__init__(*args, **kwargs)
        self.model = FloorModel(self.model_args)

class Target(Base):
    def __init__(self, *args, **kwargs):
        super(Target, self).__init__(*args, **kwargs)
        self.model = TargetModel(self.model_args)

class Drone(Base):
    def __init__(self, *args, **kwargs):
        super(Drone, self).__init__(*args, **kwargs)
        self.model = DroneModel(self.model_args)

class Finger(Base):
    def __init__(self, *args, **kwargs):
        super(Finger, self).__init__(*args, **kwargs)
        self.model = FingerModel(self.model_args)
        
class Gripper(Base):
    def __init__(self, *args, **kwargs):
        super(Gripper, self).__init__(*args, **kwargs)
        self.model = GripperModel(self.model_args)

class DroneGripper(Base):
    def __init__(self, *args, **kwargs):
        super(DroneGripper, self).__init__(*args, **kwargs)
        self.model = DroneGripper(self.model_args)