# register.py
import os
import sys


_the_user_function = None

def register_processor(func):
    global _the_user_function
    
    if _the_user_function is not None:
        print(f"Warning: Processor '{_the_user_function.__name__}' is being overridden by '{func.__name__}'")
    _the_user_function = func
    print(f"Successfully registered processor: {func.__name__}")
    
    return func