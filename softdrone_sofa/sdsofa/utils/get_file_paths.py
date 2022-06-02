import importlib_resources as pkg_resources
from sdsofa import configs, mesh

def get_config_file_path(file_name):
    with pkg_resources.path(configs, file_name) as p:
        return str(p)

def get_mesh_file_path(file_name):
    with pkg_resources.path(mesh, file_name) as p:
        return str(p)