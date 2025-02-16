# coding: UTF-8
# This is creaed 2023/12/23 by Y. Shinohara
# This is lastly modified 20xx/yy/zz by Y. Shinohara
from collections import namedtuple, OrderedDict

def dict_to_namedtuple(class_name, input_dict):
    # Create an ordered dictionary from the input dictionary
    ordered_dict = OrderedDict(sorted(input_dict.items()))

    # Create a namedtuple class dynamically
    namedtuple_class = namedtuple(class_name, ordered_dict.keys())

    # Create an instance of the namedtuple using the ordered dictionary values
    namedtuple_instance = namedtuple_class(*ordered_dict.values())

    return namedtuple_instance
