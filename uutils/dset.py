"""tools for datasets"""
import csv
import json
import os

import pandas as pd

home = os.path.expanduser("~")
root = os.path.join(home, "CS/.datasets/IMAGENET")

def build_synset_map(path=root):
    """make a map for converting id to label"""

    with open(os.path.join(path, "LOC_synset_mapping.txt"), "r") as file:
        data = file.read().split("\n")

    data = [d.split(" ") for d in data]
    data = [{"id": d[0], "label": " ".join(d[1:])} for d in data]

    id2l = {id: l for id, l in zip([d["id"] for d in data], [d["label"] for d in data])}
    l2id = {l: id for id, l in zip([d["id"] for d in data], [d["label"] for d in data])}
    data = {"id2l": id2l, "l2id": l2id}

    with open(os.path.join(path, "synset_map.json"), "w") as file:
        json.dump(data, file)


def get_synset_map(path=root):
    with open(os.path.join(path, "synset_map.json"), "r") as file:
        return json.load(file)


def build_solution_map(path=root):
    with open(os.path.join(path, "LOC_val_solution.csv"), "r") as file:
        data = file.read().split("\n")[1:]

    data = [d.split(",") for d in data if d]
    paths = [d[0] for d in data]
    values = [d[1] for d in data]
    values = [v.split(" ") for v in values]
    values = [
        {"labels": [i for i in v if "n" in i], "boxes": [i for i in v if not "n" in i]}
        for v in values
    ]
    for v in values:
        b = v["boxes"]
        v["boxes"] = [[b[i], b[i + 1], b[i + 2], b[i + 3]] for i in range(len(b) - 3)]

    data = {k: v for k, v in zip(paths, values)}

    with open(os.path.join(path, "solution_map.json"), "w") as file:
        json.dump(data, file)


def get_solution_map(path=root):
    with open(os.path.join(path, "solution_map.json"), "r") as file:
        return json.load(file)

def write_eval(*, name, path=root, acc=None, dt=None):
    assert (acc or dt)
    data = {'accuracy':acc, 'dt':dt}
    with open(os.path.join(path,'results',f'{name}.json'),'w') as file:
        json.dump(data,file)

def read_eval(*, name, path=root):
    with open(os.path.join(path,'results',f'{name}.json'),'r') as file:
        return json.load(file)

