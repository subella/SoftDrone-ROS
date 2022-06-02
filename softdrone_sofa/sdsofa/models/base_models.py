from stlib.physics.rigid import RigidObject

class BaseRigidObject(object):
	def __init__(self, root_node, **kwargs):
		print kwargs
		self.object = RigidObject(root_node, **kwargs)

class Floor(BaseRigidObject):
	def __init__(self, root_node, **kwargs):
		super(Floor, self).__init__(root_node, **kwargs)

class Target(BaseRigidObject):
	def __init__(self, root_node, **kwargs):
		super(Target, self).__init__(root_node, **kwargs)

class Drone(BaseRigidObject):
	def __init__(self, root_node, **kwargs):
		super(Drone, self).__init__(root_node, **kwargs)