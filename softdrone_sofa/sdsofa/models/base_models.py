from stlib.physics.rigid import RigidObject
from stlib.physics.deformable import ElasticMaterialObject

class BaseRigidObject(object):
	def __init__(self, root_node, **kwargs):
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

class BaseElasticObject(object):
	def __init__(self, root_node, **kwargs):
		self.eobject = ElasticMaterialObject(root_node, **kwargs)

class Finger(BaseElasticObject):
	def __init__(self, root_node, **kwargs):
		super(Finger, self).__init__(root_node, **kwargs)

class Gripper(BaseElasticObject):
	def __init__(self, root_node, **kwargs):
		pass

	# eobject = ElasticMaterialObject(node,
 #                                        name='eobject',
 #                                        volumeMeshFileName=volumeMeshFileName,
 #                                        surfaceMeshFileName=surfaceMeshFileName,
 #                                        scale=scale,
 #                                        surfaceColor=color,
 #                                        poissonRatio=self.v,
 #                                        youngModulus=self.E,
 #                                        # totalMass=mass,
 #                                        density=self.rho,
 #                                        rotation=rotation,
 #                                        translation=translation,
 #                                        collisionGroup=[1])