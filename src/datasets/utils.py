'''
 # @ Author: Zhi Wu
 # @ Create Time: 1970-01-01 00:00:00
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-08-27 15:52:34
 # @ Description: Dataset loader.
 '''

def build_dataset(name, **kwargs):
    from . import build_minist_dataset
    _ = {
        'minist' : build_minist_dataset,
    }
    assert name in _.keys(), f'Unknown dataset {name}'
    return _[name.lower()](**kwargs)
