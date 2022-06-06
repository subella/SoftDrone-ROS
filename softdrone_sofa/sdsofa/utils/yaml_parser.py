import yaml
from sdsofa.models.base_models import Floor, Target, Drone, Finger
from sdsofa.utils.utils import abs_path

def read_yaml_to_dict(config_file):
    config_file_path = abs_path(config_file)
    with open(config_file_path, 'r') as f:
        try:
            params = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    return params

def parse_rigid_kwargs(sub_params, kwargs):
    if "scale" in sub_params.keys():
        kwargs["uniformScale"] = sub_params["scale"]
    if "volume" in sub_params.keys():
        kwargs["volume"] = sub_params["volume"]
    if "inertia_matrix" in sub_params.keys():
        kwargs["inertiaMatrix"] = sub_params["inertia_matrix"]
    if "color" in sub_params.keys():
        kwargs["color"] = sub_params["color"]
    if "contact_friction" in sub_params.keys():
        kwargs["contactFriction"] = sub_params["contact_friction"]
    if "is_static" in sub_params.keys():
        kwargs["isAStaticObject"] = sub_params["is_static"]
    return kwargs

def parse_elastic_kwargs(sub_params, kwargs):
    if "volume_mesh_file_name" in sub_params.keys():
        kwargs["volumeMeshFileName"] = abs_path(sub_params["volume_mesh_file_name"])
    if "collision_mesh_file_name" in sub_params.keys():
        kwargs["collisionMesh"] = abs_path(sub_params["collision_mesh_file_name"])
    if "scale" in sub_params.keys():
        kwargs["scale"] = sub_params["scale"]
    if "color" in sub_params.keys():
        kwargs["surfaceColor"] = sub_params["color"]
    if "poisson_ratio" in sub_params.keys():
        kwargs["poissonRatio"] = sub_params["poisson_ratio"]
    if "young_modulus" in sub_params.keys():
        kwargs["youngModulus"] = sub_params["young_modulus"]
    # TODO: it seems the built in stlib prefab doesn't use density
    # if "density" in sub_params.keys():
    #     kwargs["density"] = sub_params["density"]
    if "collision_group" in sub_params.keys():
        kwargs["collisionGroup"] = sub_params["collision_group"]
    return kwargs

def parse_common_kwargs(sub_params):
    kwargs = {}
    if "name" in sub_params.keys():
        kwargs["name"] = sub_params["name"]
    if "surface_mesh_file_name" in sub_params.keys():
        kwargs["surfaceMeshFileName"] = abs_path(sub_params["surface_mesh_file_name"])
    if "translation" in sub_params.keys():
        kwargs["translation"] = sub_params["translation"]
    if "rotation" in sub_params.keys():
        kwargs["rotation"] = sub_params["rotation"]
    if "mass" in sub_params.keys():
        kwargs["totalMass"] = sub_params["mass"]
    return kwargs

def parse_kwargs_from_params(sub_params):
    kwargs = parse_common_kwargs(sub_params)
    if sub_params["type"] == "Rigid":
        kwargs = parse_rigid_kwargs(sub_params, kwargs)
    elif sub_params["type"] == "Elastic":
        kwargs = parse_elastic_kwargs(sub_params, kwargs)
    return kwargs

def create_object(root_node, sub_params, obj_kwargs):
    if sub_params["class"] == "Floor":
        obj = Floor(root_node, **obj_kwargs)
    elif sub_params["class"] == "Target":
        obj = Target(root_node, **obj_kwargs)
    elif sub_params["class"] == "Drone":
        obj = Drone(root_node, **obj_kwargs)
    elif sub_params["class"] == "Finger":
        obj = Finger(root_node, **obj_kwargs)
    return obj

def parse_and_create_object_list(root_node, obj_params_list):
    if obj_params_list is None:
        return None
    obj_list = []
    for obj_params in obj_params_list:
        sub_params = read_yaml_to_dict(obj_params["config_file"])
        # Add translation and positions to sub_params.
        sub_params["name"] = obj_params["name"]
        sub_params["translation"] = obj_params["translation"]
        sub_params["rotation"] = obj_params["rotation"]
        obj_kwargs = parse_kwargs_from_params(sub_params)
        obj = create_object(root_node, sub_params, obj_kwargs)
        obj_list.append(obj)
    return obj_list

def create_scene_from_yaml(root_node, config_file):
    params = read_yaml_to_dict(config_file)
	# Configurable parameters
    root_node.dt = params.get("timestep", 0.01)

    obj_params_list = params.get("objects", None)
    obj_list = parse_and_create_object_list(root_node, obj_params_list)