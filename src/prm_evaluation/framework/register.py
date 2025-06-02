# register.py
import os
import sys
from typing import Callable, Dict, Optional


_registered_functions: Dict[str, Callable] = {}

def register_processor(name: str = None):
    def decorator(func: Callable):
        nonlocal name
        name = name or func.__name__
        
        if name in _registered_functions:
            print(f"Warning: Processor '{name}' is being overridden "
                  f"(Origin: {_registered_functions[name].__name__})")
        
        _registered_functions[name] = func
        print(f"Successfully registered processor: {name}")
        return func
    return decorator

def get_processor(name: str) -> Callable:
    return _registered_functions.get(name)

def list_processors() -> Dict[str, Callable]:
    return _registered_functions.copy()