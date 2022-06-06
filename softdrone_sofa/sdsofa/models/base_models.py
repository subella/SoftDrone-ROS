from stlib.physics.rigid import RigidObject
from stlib.physics.deformable import ElasticMaterialObject
from sdsofa.utils.transforms import Pose, child_to_world

class Scene(object):
	def __init__(self, root_node):
	    scene = Scene(root_node, plugins=["SoftRobots",
	                                     "SofaSparseSolver"])
	    root_node.createObject("BackgroundSetting", color=[1., 1., 1.])
	    root_node.gravity = [0., -9.81, 0]
	    root_node.createObject("FreeMotionAnimationLoop")
	    root_node.createObject("GenericConstraintSolver", maxIterations=1e3, tolerance=1e-8)
	    ContactHeader(root_node, alarmDistance=1, contactDistance=.75, frictionCoef=.2)
	    root_node.createObject("LightManager")
	    root_node.createObject("DirectionalLight", color=[1, 1, 1], direction=[0, -1, 0])

class BaseObject(object):
	def __init__(self, 
				 root_node, 
				 base_kwargs,
				 object_kwargs):
		self.root_node = root_node
		self.base_kwargs = base_kwargs
		self.object_kwargs = object_kwargs
		print self.base_kwargs
		print self.object_kwargs
		self.object = None
		self.convert_relative_pose_to_world_frame()

	def convert_relative_pose_to_world_frame(self):
		pos_wrt_parent = self.object_kwargs["translation"]
		rot_wrt_parent = self.object_kwargs["rotation"]
		parent_pos_wrt_world = self.base_kwargs["parent_pos_wrt_world"]
		parent_rot_wrt_world = self.base_kwargs["parent_rot_wrt_world"]
		pose_wrt_parent = Pose(pos_wrt_parent, rot_wrt_parent)
		parent_pose_wrt_world = Pose(parent_pos_wrt_world, parent_rot_wrt_world)
		translation, rotation = child_to_world(pose_wrt_parent, parent_pose_wrt_world)
		self.object_kwargs["translation"] = translation
		self.object_kwargs["rotation"] = rotation
		print translation
		print rotation

	def create(self):
		pass

class BaseRigidObject(BaseObject):
	def __init__(self, *args, **kwargs):
		super(BaseRigidObject, self).__init__(*args, **kwargs)
		self.object = RigidObject(self.root_node, **self.object_kwargs)

	# def create(self):
	# 	self.object = RigidObject(self.root_node, **self.kwargs)

class Floor(BaseRigidObject):
	def __init__(self, *args, **kwargs):
		super(Floor, self).__init__(*args, **kwargs)

class Target(BaseRigidObject):
	def __init__(self, *args, **kwargs):
		super(Target, self).__init__(*args, **kwargs)

class Drone(BaseRigidObject):
	def __init__(self, *args, **kwargs):
		super(Drone, self).__init__(*args, **kwargs)

class BaseElasticObject(BaseObject):
	def __init__(self, *args, **kwargs):
		super(BaseElasticObject, self).__init__(*args, **kwargs)
		self.object = ElasticMaterialObject(self.root_node, **self.object_kwargs)

	# def create(self):
	# 	self.object = ElasticMaterialObject(self.root_node, **self.kwargs)

class Finger(BaseElasticObject):
	def __init__(self, *args, **kwargs):
		super(Finger, self).__init__(*args, **kwargs)

class Gripper(BaseElasticObject):
	def __init__(self, 
			     root_node,
			     translation = [],
			     rotation = [],
				 finger_array = [],
				 ):
		pass