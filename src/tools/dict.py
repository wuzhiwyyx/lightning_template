'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-25 00:45:25
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-25 00:46:29
 # @ Description: Utilities of dict.
 '''


from copy import deepcopy


def deep_update(raw, new):
    if new is None:
        return raw
    foo = deepcopy(raw)
    if isinstance(new, list):
        foo = [deep_update(foo, n) for n in new]
    else:
        update_keys(foo, new)
        insert_keys(foo, new)
    return foo

def update_keys(raw, new):
    for key in raw:
        if key not in new.keys():
            continue
        if isinstance(raw[key], dict) and isinstance(new[key], dict):
            raw[key] = deep_update(raw[key], new[key])
        else:
            raw[key] = new[key]

def insert_keys(raw, new):
    update_dict = {}
    for key in new:
        if key not in raw.keys():
            update_dict[key] = new[key]
    raw.update(update_dict)
