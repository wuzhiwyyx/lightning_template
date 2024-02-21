'''
 # @ Author: Zhi Wu
 # @ Create Time: 1970-01-01 00:00:00
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-08-27 14:20:31
 # @ Description: Functions of load and parse config file.
 '''

from yaml import Loader, load
from copy import deepcopy

class ConfigDict(dict):
    """
    Access-by-attribute, case-insensitive dictionary
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            if isinstance(v, dict) and not isinstance(v, ConfigDict):
                self[k] = ConfigDict(v)
        # for k, v in list(self.items()):
        #     if not k.islower() and k.isupper():
        #         self.pop(k)
        #         self[k.lower()] = v
        #     if isinstance(v, dict) and not isinstance(v, ConfigDict):
        #         if not k.islower() and k.isupper():
        #             self[k.lower()] = ConfigDict(v)
        #         else:
        #             self[k] = ConfigDict(v)

    def __getattr__(self, name):
        return self.get(name.lower())

    def __setattr__(self, name, value):
        self.__setitem__(name.lower(), value)
    
    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, ConfigDict) else v for k, v in self.items()}

    # ====================================================
    # ----------------------------------------------------
    # this code block is used to make ConfigDict deep copyable
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        return ConfigDict(deepcopy(dict(self), memo=memo))
    # ----------------------------------------------------
    # ====================================================
    

    # ====================================================
    # ----------------------------------------------------
    # this code block is used to make ConfigDict pickable
    def __getstate__(self): 
        return self.__dict__

    def __setstate__(self, d): 
        self.__dict__.update(d)
    # ----------------------------------------------------
    # ====================================================



def load_config(config_file):
    with open(config_file, 'r') as f:
        cfgs = ConfigDict(load(f, Loader=Loader))
    return cfgs