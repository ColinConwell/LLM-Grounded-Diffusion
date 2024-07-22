from warnings import filterwarnings
from os import environ

def silence_tensorflow():
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def _is_jupyter():
    return any('JPY' in var for var in environ)

def ignore_warnings(**kwargs):
    filterwarnings('ignore', **kwargs)

silence_tensorflow()

if _is_jupyter():
    ignore_warnings()