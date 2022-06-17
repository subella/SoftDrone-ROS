import os
import json

def abs_path(relative_path):
	abs_path = os.path.join(os.environ['ROOT_PATH'], relative_path)
	return abs_path

def parse_json(file_path):
	f = open(file_path)
	data = json.load(f)
	f.close()
	return data