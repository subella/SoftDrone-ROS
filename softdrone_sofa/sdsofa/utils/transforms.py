import numpy as np
from scipy.spatial.transform import Rotation as R


class Pose(object):
	def __init__(self, translation=[0,0,0], rotation=[0,0,0]):
		self.translation = translation
		self.rotation = rotation

	def make_homo(self):
		r = np.array(R.from_euler('xyz', self.rotation, degrees=True).as_dcm())
		t = np.array(self.translation).reshape(3,1)
		T = np.hstack((r, t))
		T = np.vstack((T, [0,0,0,1]))
		return T

	def update_from_homo(self, T):
		r = T[:3,:3]
		t = T[:3,3]

		self.translation = t.T
		self.rotation = R.from_matrix(r).as_euler('zyx')

def child_to_world(child_wrt_parent, parent_wrt_world):
	print parent_wrt_world.make_homo()
	print child_wrt_parent.make_homo()
	child_wrt_world = parent_wrt_world.make_homo().dot(child_wrt_parent.make_homo())
	r = child_wrt_world[:3,:3]
	t = child_wrt_world[:3,3]

	translation = t.T
	rotation = R.from_dcm(r).as_euler('xyz', degrees=True)
	return translation, rotation
