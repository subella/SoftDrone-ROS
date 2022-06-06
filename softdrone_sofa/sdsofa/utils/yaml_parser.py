import yaml
from sdsofa.models.base_models import Floor, Target, Drone, Finger
from sdsofa.utils.utils import abs_path

class FileParser(object):
    def __init__(self, config_file):
        self.sub_params = self.parse(config_file)
        # self.child_parsers = create_children_parsers(keyword)

    def parse(self, config_file):
        config_file_path = abs_path(config_file)
        print config_file_path
        with open(config_file_path, 'r') as f:
            try:
                params = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)
        return params
        
class YAMLParser(FileParser):
    def __init__(self, root_node, config_file, top_params=None):
        super(YAMLParser, self).__init__(config_file)
        self.root_node = root_node
        self.top_params = top_params
        self.children = []
        if "objects" in self.sub_params:
            for child_params in self.sub_params["objects"]:
                if child_params["type"] == "Floor":
                    ChildParser = FloorParser
                elif child_params["type"] == "Target":
                    ChildParser = TargetParser
                elif child_params["type"] == "Drone":
                    ChildParser = DroneParser
                elif child_params["type"] == "Finger":
                    ChildParser = FingerParser
                elif child_params["type"] == "Gripper":
                    ChildParser = GripperParser
                else:
                    continue
                child_parser = ChildParser(root_node, child_params["config_file"], top_params, child_params)
                self.children.append(child_parser)

        def format_kwargs(self):
            pass

        def create_object(self):
            pass

class SceneParser(YAMLParser):
    def __init__(self, root_node, config_file, top_params=None):
        super(SceneParser, self).__init__(root_node, config_file, top_params)

class ObjectParser(YAMLParser):
    def __init__(self, root_node, config_file, parent_params=None, top_params=None):
        super(ObjectParser, self).__init__(root_node, config_file, top_params)
        self.parent_params = parent_params
        self.object = None
        self.base_kwargs = {}
        self.object_kwargs = {}
        self.format_kwargs()
        self.create_object()

    def format_kwargs(self):
        # Parent Params
        if self.parent_params is not None:
            if "translation" in self.parent_params.keys():
                self.base_kwargs["parent_pos_wrt_world"] = self.parent_params["translation"]
            if "rotation" in self.parent_params.keys():
                self.base_kwargs["parent_rot_wrt_world"] = self.parent_params["rotation"]
        else:
            self.base_kwargs["parent_pos_wrt_world"] = [0,0,0]
            self.base_kwargs["parent_rot_wrt_world"] = [0,0,0]

        # Top Level Params
        if "name" in self.top_params.keys():
            self.object_kwargs["name"] = self.top_params["name"]
        if "translation" in self.top_params.keys():
            self.object_kwargs["translation"] = self.top_params["translation"]
        if "rotation" in self.top_params.keys():
            self.object_kwargs["rotation"] = self.top_params["rotation"]

    def create_object(self):
        pass

class RigidParser(ObjectParser):
    def __init__(self, *args, **kwargs):
        super(RigidParser, self).__init__(*args, **kwargs)

    def format_kwargs(self):
        super(RigidParser, self).format_kwargs()
        # Sub Level Params
        if "surface_mesh_file_name" in self.sub_params.keys():
            self.object_kwargs["surfaceMeshFileName"] = abs_path(self.sub_params["surface_mesh_file_name"])
        if "mass" in self.sub_params.keys():
            self.object_kwargs["totalMass"] = self.sub_params["mass"]
        if "scale" in self.sub_params.keys():
            self.object_kwargs["uniformScale"] = self.sub_params["scale"]
        if "volume" in self.sub_params.keys():
            self.object_kwargs["volume"] = self.sub_params["volume"]
        if "inertia_matrix" in self.sub_params.keys():
            self.object_kwargs["inertiaMatrix"] = self.sub_params["inertia_matrix"]
        if "color" in self.sub_params.keys():
            self.object_kwargs["color"] = self.sub_params["color"]
        if "contact_friction" in self.sub_params.keys():
            self.object_kwargs["contactFriction"] = self.sub_params["contact_friction"]
        if "is_static" in self.sub_params.keys():
            self.object_kwargs["isAStaticObject"] = self.sub_params["is_static"]

    def create_object(self):
        self.object = BaseRigidObject(self.root_node, self.base_kwargs, self.object_kwargs)

class FloorParser(RigidParser):
    def __init__(self, *args, **kwargs):
        super(FloorParser, self).__init__(*args, **kwargs)

    def create_object(self):
        self.object = Floor(self.root_node, self.base_kwargs, self.object_kwargs)

class TargetParser(RigidParser):
    def __init__(self, *args, **kwargs):
        super(TargetParser, self).__init__(*args, **kwargs)

    def create_object(self):
        self.object = Target(self.root_node, self.base_kwargs, self.object_kwargs)

class DroneParser(RigidParser):
    def __init__(self, *args, **kwargs):
        super(DroneParser, self).__init__(*args, **kwargs)

    def create_object(self):
        self.object = Drone(self.root_node, self.base_kwargs, self.object_kwargs)

class ElasticParser(ObjectParser):
    def __init__(self, *args, **kwargs):
        super(ElasticParser, self).__init__(*args, **kwargs)

    def format_kwargs(self):
        super(ElasticParser, self).format_kwargs()
        if "mass" in self.sub_params.keys():
            self.object_kwargs["totalMass"] = self.sub_params["mass"]
        if "surface_mesh_file_name" in self.sub_params.keys():
            self.object_kwargs["surfaceMeshFileName"] = abs_path(self.sub_params["surface_mesh_file_name"])
        if "volume_mesh_file_name" in self.sub_params.keys():
            self.object_kwargs["volumeMeshFileName"] = abs_path(self.sub_params["volume_mesh_file_name"])
        if "collision_mesh_file_name" in self.sub_params.keys():
            self.object_kwargs["collisionMesh"] = abs_path(self.sub_params["collision_mesh_file_name"])
        if "scale" in self.sub_params.keys():
            self.object_kwargs["scale"] = self.sub_params["scale"]
        if "color" in self.sub_params.keys():
            self.object_kwargs["surfaceColor"] = self.sub_params["color"]
        if "poisson_ratio" in self.sub_params.keys():
            self.object_kwargs["poissonRatio"] = self.sub_params["poisson_ratio"]
        if "young_modulus" in self.sub_params.keys():
            self.object_kwargs["youngModulus"] = self.sub_params["young_modulus"]
        # TODO: it seems the built in stlib prefab doesn't use density
        # if "density" in self.sub_params.keys():
        #     self.object_kwargs["density"] = self.sub_params["density"]
        if "collision_group" in self.sub_params.keys():
            self.object_kwargs["collisionGroup"] = self.sub_params["collision_group"]

        def create_object(self):
            self.object = BaseElasticObject(self.root_node, self.base_kwargs, self.object_kwargs)

class FingerParser(ElasticParser):
    def __init__(self, *args, **kwargs):
        super(FingerParser, self).__init__(*args, **kwargs)

    def create_object(self):
        self.object = Finger(self.root_node, self.base_kwargs, self.object_kwargs)

class GripperParser(ObjectParser):
    def __init__(self, *args, **kwargs):
        super(GripperParser, self).__init__(*args, **kwargs)

    # def format_kwargs(self):
    #     for finger in self.children:
    #         print finger.top_params
    #     # print self.top_params
    #     # print self.sub_params

    # def create_object(self):
    #     self.object


def create_scene_from_yaml(root_node, config_file):
    scene_parser = SceneParser(root_node, config_file)

 #    params = read_yaml_to_dict(config_file)
	# # Configurable parameters
 #    root_node.dt = params.get("timestep", 0.01)

 #    obj_params_list = params.get("objects", None)
 #    obj_list = parse_and_create_object_list(root_node, obj_params_list)