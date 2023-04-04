import argparse
import json


def get_params(json_file=None):
    with open(json_file) as f:
        summary_dict = json.load(fp=f)
    args = argparse.Namespace(**summary_dict)
    return args
