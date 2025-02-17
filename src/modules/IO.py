# coding: UTF-8
# This is creaed 2023/12/22 by Y. Shinohara
# This is lastly modified 20xx/yy/zz by Y. Shinohara
import sys
import yaml
from modules.constants import *
import numpy as np

class Input:
    """A class for input."""

    default_parameter_file = './default_parameters.yaml'

    @classmethod
    def read_input(cls):
        # Read the default input
        with open(cls.default_parameter_file) as file:
            def_param = yaml.safe_load(file)
        param = dict(def_param)

        argv = sys.argv
        argc = len(argv)

        # Reading input file name from the standard input
        if argc == 1:
            print('# The default parameters are chosen.')
        elif argc == 2:
            print('# Name of the input file is "' + argv[1] + '".')
            with open(argv[1]) as file:
                input_param = yaml.safe_load(file)
                cls.update_parameters(param, input_param)
        else:
            cls.print_error_and_exit('ERROR: Incorrect number of arguments. Expected 1 or 2.')

        cls.print_parameters(param)
        
        return param 

    @staticmethod
    def update_parameters(default_param, input_param):
        keys = ['unit', 'model', 'kgrid', 'tgrid', 'field', 'tevolution', 'options']
        for key in keys:
            if key in input_param:
                param2 = input_param[key]
                default_param[key].update(param2)
                print(f'# `{key}` key is in the input.')

    @staticmethod
    def print_parameters(param):
        print('#=====Print the parameters')
        for key, value in param.items():
            print(f'# {key} : {value}')

    @staticmethod
    def print_error_and_exit(message):
        print(message)
        sys.exit()
