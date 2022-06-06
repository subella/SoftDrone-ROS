import os

def abs_path(relative_path):
	abs_path = os.path.join(os.environ['ROOT_PATH'], relative_path)
	return abs_path