import yaml
from sdsofa.models.base_models import Floor, Target, Drone
from sdsofa.utils.get_file_paths import get_config_file_path, get_mesh_file_path

def read_yaml_to_dict(config_file):
    config_file_path = get_config_file_path(config_file)
    with open(config_file_path, 'r') as f:
        try:
            params = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    return params

def parse_and_create_object_list(root_node, obj_dict_list, Object):
    if obj_dict_list is None:
        return None
    obj_list = []
    for obj_dict in obj_dict_list:
        obj_params = read_yaml_to_dict(obj_dict["config_file"])
        obj_dict.update(obj_params)
        # Delete keys not in kwargs.
        del obj_dict["config_file"]
        # Update param to point to actual path.
        obj_dict["surfaceMeshFileName"] = get_mesh_file_path(obj_dict["surfaceMeshFileName"])
        obj = Object(root_node, **obj_dict)
        obj_list.append(obj)
    return obj_list

def create_scene_from_yaml(root_node, config_file):
    params = read_yaml_to_dict(config_file)
	# Configurable parameters
    root_node.dt = params.get("timestep", 0.01)

    floors_dict = params.get("floors", None)
    floors = parse_and_create_object_list(root_node, floors_dict, Floor)

    targets_dict = params.get("targets", None)
    targets = parse_and_create_object_list(root_node, targets_dict, Target)

    drones_dict = params.get("drones", None)
    drones = parse_and_create_object_list(root_node, drones_dict, Drone)