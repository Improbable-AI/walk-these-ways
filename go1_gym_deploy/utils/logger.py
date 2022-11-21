import copy
import pickle as pkl

import numpy as np
import torch


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            print(key)
            element = class_to_dict(val)
        result[key] = element
    return result


class MultiLogger:
    def __init__(self):
        self.loggers = {}

    def add_robot(self, name, cfg):
        print(name, cfg)
        self.loggers[name] = EpisodeLogger(cfg)

    def log(self, name, info):
        self.loggers[name].log(info)

    def save(self, filename):
        with open(filename, 'wb') as file:
            logdict = {}
            for key in self.loggers.keys():
                logdict[key] = [class_to_dict(self.loggers[key].cfg), self.loggers[key].infos]
            pkl.dump(logdict, file)
            print(f"Saved log! Number of timesteps: {[len(self.loggers[key].infos) for key in self.loggers.keys()]}; Path: {filename}")

    def read_metric(self, metric, robot_name=None):
        if robot_name is None:
            robot_name = list(self.loggers.keys())[0]
        logger = self.loggers[robot_name]

        metric_arr = []
        for info in logger.infos:
            metric_arr += [info[metric]]
        return np.array(metric_arr)

    def reset(self):
        for key, log in self.loggers.items():
            log.reset()


class EpisodeLogger:
    def __init__(self, cfg):
        self.infos = []
        self.cfg = cfg

    def log(self, info):
        for key in info.keys():
            if isinstance(info[key], torch.Tensor):
                info[key] = info[key].detach().cpu().numpy()

            if isinstance(info[key], dict):
                continue
            elif "image" not in key:
                info[key] = copy.deepcopy(info[key])

        self.infos += [dict(info)]

    def reset(self):
        self.infos = []
