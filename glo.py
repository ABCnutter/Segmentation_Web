# -*- coding: utf-8 -*-

def _init():  # 初始化
    global _global_dict
    _global_dict = {}

def print_glo():
    from pprint import pprint
    pprint(_global_dict)
# def set_value(key, value):
#     # 定义一个全局变量
#     _global_dict[key] = value

# def get_value(key):
#     # 获得一个全局变量，不存在则提示读取对应变量失败
#     try:
#         return _global_dict[key]
#     except:
#         print('read ' + key + 'failed\r\n')

def set_value(keys, value, current_dict=None):
    # 定义一个全局变量
    global _global_dict

    # 如果 current_dict 为空，则从全局变量开始
    if current_dict is None:
        current_dict = _global_dict

    # 获取当前层级的键
    key = keys[0]

    if len(keys) == 1:
        # 单层赋值
        if isinstance(current_dict.get(key), set):
            current_dict[key].add(value)
        else:
            current_dict[key] = value  # 创建一个新的集合
    else:
        # 多层赋值
        if key not in current_dict:
            current_dict[key] = {}
        set_value(keys[1:], value, current_dict[key])


def get_value(keys, current_dict=None):
    # 定义一个全局变量
    global _global_dict

    # 如果 current_dict 为空，则从全局变量开始
    if current_dict is None:
        current_dict = _global_dict

    # 获取当前层级的键
    key = keys[0]

    if len(keys) == 1:
        # 单层获取值
        return current_dict.get(key)
    else:
        # 多层获取值
        if key not in current_dict:
            return None
        return get_value(keys[1:], current_dict[key])


def remove_setvalue(keys, value, current_dict=None):
    # 定义一个全局变量
    global _global_dict

    # 如果 current_dict 为空，则从全局变量开始
    if current_dict is None:
        current_dict = _global_dict

    # 获取当前层级的键
    key = keys[0]

    if len(keys) == 1:
        # 单层移除值
        if isinstance(current_dict.get(key), set):
            current_dict[key].discard(value)
    else:
        # 多层移除值
        if key not in current_dict:
            return
        remove_setvalue(keys[1:], value, current_dict[key])

def delete_key(keys, current_dict=None):
    # 定义一个全局变量
    global _global_dict

    # 如果 current_dict 为空，则从全局变量开始
    if current_dict is None:
        current_dict = _global_dict

    # 获取当前层级的键
    key = keys[0]

    if len(keys) == 1:
        # 单层删除键值对
        current_dict.pop(key, None)
    else:
        # 多层删除键值对
        if key not in current_dict:
            return
        delete_key(keys[1:], current_dict[key])

from enum import Enum
import json

def save_global_variables(file_path, key):
    global _global_dict
    courrect_dict = _global_dict[key]
    
    def convert_to_serializable(value):
        if isinstance(value, set):
            return list(value)
        elif isinstance(value, Enum):
            return value.value
        return value
    
    with open(file_path, 'w') as file:
        json.dump(courrect_dict, file, indent=2, default=convert_to_serializable)

if __name__ == "__main__":
    _init()
    set_value(['f'], set())
    set_value(['f'], 'sasdasdas')
    set_value(['f'], 'dfsvsvsdf')
    set_value(['f'], 'qrrewrdfsd')
    print('dfsvsvsdf' in get_value(['f']))
    print(get_value(['f']))
    
    remove_setvalue(['f'], 'sasdasdas')
    print('sasdasdas' in get_value(['f']))
    print(get_value(['f']))

    # set_value(['a', 'b', 'c'], 1)
    set_value(['a', 'e'], 2)
    set_value(['a', 'f'], 3)
    set_value(['a', 'c'], 3)
    set_value(['a', 'q'], 3)
    set_value(['a', 'w'], 3)
    set_value(['a', 'fr'], 3)

    # print(get_value(['a', 'b', 'c']))
    # print(get_value(['a', 'b']))
    print(get_value(['a']))
    set_value(['c', 'd'], 30)
    set_value(['c', 'e'], 40)

    print(get_value(['c']))
    delete_key('c')
    print(get_value(['c']))
