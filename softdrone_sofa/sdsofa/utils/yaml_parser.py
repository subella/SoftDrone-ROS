import yaml
from sdsofa.classes import Floor, Target, Drone, Finger, Gripper, DroneGripper
from sdsofa.utils.utils import abs_path
from sdsofa.utils.transforms import make_tf_list

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
    def __init__(self, node, config_file, parent=None, top_params=None):
        super(YAMLParser, self).__init__(config_file)
        self.node = node
        self.top_params = top_params
        self.parent = parent
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
                elif child_params["type"] == "DroneGripper":
                    ChildParser = DroneGripperParser
                else:
                    continue
                child_node = self.node.createChild(child_params["name"])
                child_parser = ChildParser(child_node, 
                                           child_params["config_file"], 
                                           parent=self, 
                                           top_params=child_params)
                self.children.append(child_parser)

class SceneParser(YAMLParser):
    def __init__(self, *args, **kwargs):
        super(SceneParser, self).__init__(*args, **kwargs)
        self.objects = [child.object for child in self.children]

class ObjectParser(YAMLParser):
    def __init__(self, *args, **kwargs):
        super(ObjectParser, self).__init__(*args, **kwargs)
        self.object = None
        self.model_args = {}
        self.controller_args = {}
        self.format_kwargs()
        self.create_object()

    def format_kwargs(self):
        # Top Level Params
        
        if "name" in self.top_params.keys():
            self.model_args["name"] = self.top_params["name"]
        if "translation" in self.top_params.keys():
            self.model_args["local_translation"] = self.top_params["translation"]
        if "rotation" in self.top_params.keys():
            self.model_args["local_rotation"] = self.top_params["rotation"]

        self.model_args["node"] = self.node
        self.model_args["tf_list"] = make_tf_list([], self)

    def create_object(self):
        pass

class RigidParser(ObjectParser):
    def __init__(self, *args, **kwargs):
        super(RigidParser, self).__init__(*args, **kwargs)

    def format_kwargs(self):
        super(RigidParser, self).format_kwargs()
        self.model_args["material_args"] = {}
        self.model_args["material_args"]["node"] = self.node
        if "name" in self.top_params.keys():
            self.model_args["material_args"]["name"] = self.top_params["name"]
        # Sub Level Params
        if "surface_mesh_file_name" in self.sub_params.keys():
            self.model_args["material_args"]["surfaceMeshFileName"] = abs_path(self.sub_params["surface_mesh_file_name"])
        if "mass" in self.sub_params.keys():
            self.model_args["material_args"]["totalMass"] = self.sub_params["mass"]
        if "scale" in self.sub_params.keys():
            self.model_args["material_args"]["uniformScale"] = self.sub_params["scale"]
        if "volume" in self.sub_params.keys():
            self.model_args["material_args"]["volume"] = self.sub_params["volume"]
        if "inertia_matrix" in self.sub_params.keys():
            self.model_args["material_args"]["inertiaMatrix"] = self.sub_params["inertia_matrix"]
        if "color" in self.sub_params.keys():
            self.model_args["material_args"]["color"] = self.sub_params["color"]
        if "contact_friction" in self.sub_params.keys():
            self.model_args["material_args"]["contactFriction"] = self.sub_params["contact_friction"]
        if "is_static" in self.sub_params.keys():
            self.model_args["material_args"]["isAStaticObject"] = self.sub_params["is_static"]

class FloorParser(RigidParser):
    def __init__(self, *args, **kwargs):
        super(FloorParser, self).__init__(*args, **kwargs)

    def create_object(self):
        self.object = Floor(self.model_args, 
                            self.controller_args)

class TargetParser(RigidParser):
    def __init__(self, *args, **kwargs):
        super(TargetParser, self).__init__(*args, **kwargs)

    def create_object(self):
        self.object = Target(self.model_args, 
                             self.controller_args)

class DroneParser(RigidParser):
    def __init__(self, *args, **kwargs):
        super(DroneParser, self).__init__(*args, **kwargs)

    def create_object(self):
        self.object = Drone(self.model_args, 
                            self.controller_args)

class ElasticParser(ObjectParser):
    def __init__(self, *args, **kwargs):
        super(ElasticParser, self).__init__(*args, **kwargs)

    def format_kwargs(self):
        super(ElasticParser, self).format_kwargs()
        self.model_args["material_args"] = {}
        self.model_args["material_args"]["attachedTo"] = self.node
        if "name" in self.top_params.keys():
            self.model_args["material_args"]["name"] = self.top_params["name"]
        if "mass" in self.sub_params.keys():
            self.model_args["material_args"]["totalMass"] = self.sub_params["mass"]
        if "surface_mesh_file_name" in self.sub_params.keys():
            self.model_args["material_args"]["surfaceMeshFileName"] = abs_path(self.sub_params["surface_mesh_file_name"])
        if "volume_mesh_file_name" in self.sub_params.keys():
            self.model_args["material_args"]["volumeMeshFileName"] = abs_path(self.sub_params["volume_mesh_file_name"])
        if "collision_mesh_file_name" in self.sub_params.keys():
            self.model_args["material_args"]["collisionMesh"] = abs_path(self.sub_params["collision_mesh_file_name"])
        if "scale" in self.sub_params.keys():
            self.model_args["material_args"]["scale"] = self.sub_params["scale"]
        if "color" in self.sub_params.keys():
            self.model_args["material_args"]["surfaceColor"] = self.sub_params["color"]
        if "poisson_ratio" in self.sub_params.keys():
            self.model_args["material_args"]["poissonRatio"] = self.sub_params["poisson_ratio"]
        if "young_modulus" in self.sub_params.keys():
            self.model_args["material_args"]["youngModulus"] = self.sub_params["young_modulus"]
        # TODO: it seems the built in stlib prefab doesn't use density
        # if "density" in self.sub_params.keys():
        #     self.model_args["material_args"]["density"] = self.sub_params["density"]
        if "collision_group" in self.sub_params.keys():
            self.model_args["material_args"]["collisionGroup"] = self.sub_params["collision_group"]

class FingerParser(ElasticParser):
    def __init__(self, *args, **kwargs):
        super(FingerParser, self).__init__(*args, **kwargs)

    def format_kwargs(self):
        super(FingerParser, self).format_kwargs()
        if "eyelet_locations_file_name" in self.sub_params.keys():
            self.model_args["eyelet_locations_file_name"] = abs_path(self.sub_params["eyelet_locations_file_name"])

    def create_object(self):
        self.object = Finger(self.model_args, 
                             self.controller_args)

class GripperParser(ObjectParser):
    def __init__(self, *args, **kwargs):
        super(GripperParser, self).__init__(*args, **kwargs)

    def format_kwargs(self):
        super(GripperParser, self).format_kwargs()
        self.model_args["fingers"] = [finger.object for finger in self.children]

    def create_object(self):
        self.object = Gripper(self.model_args, 
                              self.controller_args)

class DroneGripperParser(ObjectParser):
    def __init__(self, *args, **kwargs):
        super(DroneGripperParser, self).__init__(*args, **kwargs)

    def format_kwargs(self):
        super(DroneGripperParser, self).format_kwargs()
        for parser in self.children:
            if isinstance(parser.object, Drone):
                self.model_args["drone_model"] = parser.object.model
            elif isinstance(parser.object, Gripper):
                self.model_args["gripper_model"] = parser.object.model

    def create_object(self):
        self.object = DroneGripper(self.model_args, 
                                   self.controller_args)

def create_scene_from_yaml(root_node, config_file):
    scene_parser = SceneParser(root_node, config_file)
    return scene_parser.objects