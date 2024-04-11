'''
 # @ Author: Zhi Wu
 # @ Create Time: 2024-03-07 10:51:40
 # @ Modified by: Zhi Wu
 # @ Modified time: 2024-03-07 10:51:55
 # @ Description: Introduce register mechanism to make code more clean.
 '''


class Registry():
    _registry = {}

    @classmethod
    def register(cls, ref):
        Registry._registry[ref.__name__] = ref

    @classmethod
    def decorator(cls):
        def dec(cls):
            Registry.register(cls)
            return cls
        return dec

    @classmethod
    def register_module(cls):
        return Registry.decorator()

    @classmethod
    def register_dataset(cls):
        return Registry.decorator()
    
    @classmethod
    def register_collate(cls):
        return Registry.decorator()
    
    @classmethod
    def register_callback(cls):
        return Registry.decorator()
    
    @classmethod
    def instantiate(cls, _cls_name: str, **kwargs):
        assert _cls_name in Registry._registry, f'Class not found: {_cls_name}'
        return Registry._registry[_cls_name](**kwargs)
    
    @classmethod
    def get_class(cls, _cls_name: str):
        assert _cls_name in Registry._registry, f'Class not found: {_cls_name}'
        return Registry._registry[_cls_name]
    
    @classmethod
    def get_fn(cls, _fn_name: str):
        if _fn_name is None:
            return None
        assert _fn_name in Registry._registry, f'Function not found: {_fn_name}'
        return Registry._registry[_fn_name]
