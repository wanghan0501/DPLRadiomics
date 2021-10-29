"""
Created by Wang Han on 2018/11/19 16:32.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2018 Wang Han. SCU. All Rights Reserved.
"""
import json

import ruamel.yaml


def parse_yaml(file='cfgs/default.yaml'):
    with open(file) as f:
        return ruamel.yaml.load(f, Loader=ruamel.yaml.Loader)


def format_config(config, indent=2):
    return json.dumps(config, indent=indent, ensure_ascii=False)
