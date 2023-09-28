import numpy as np
import pandas as pd
import yaml


class Config:
    def __init__(self):
        self.config_dict = dict()

    # @staticmethod
    def get_args(self, path):
        """
        Reads hyperparameters of specified .yaml file and returns dictionary with arguments.
        :param path: The .yaml file path.
        :return: A dictionary containing all arguments.
        """

        with open(path, "r", encoding='utf-8') as stream:
            args = yaml.safe_load(stream)

        self.config_dict = args

        return self.config_dict

    @staticmethod
    def store_args(path, args):
        with open(path, "w") as stream:
            yaml.safe_dump(args, stream)


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value


def save_csv(filename, **kwargs):
    columns = []
    value_list = []
    for keys in kwargs.keys():
        columns.append(keys)
        value = kwargs[keys]
        value_list.append(value)
    value_list = np.array(value_list, dtype=np.float32).T
    dataframe = pd.DataFrame(value_list, columns=columns)
    dataframe.to_csv(filename, index=True, header=True)
