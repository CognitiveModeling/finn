import json
import jsmin
import os


class Dict(dict):
    """
    Dictionary that allows to access per attributes and to except names from being loaded
    """
    def __init__(self, dictionary: dict = None):
        super(Dict, self).__init__()

        if dictionary is not None:
            self.load(dictionary)

    def __getattr__(self, item):
        try:
            return self[item] if item in self else getattr(super(Dict, self), item)
        except AttributeError:
            raise AttributeError(f'This dictionary has no attribute "{item}"')

    def load(self, dictionary: dict, name_list: list = None):
        """
        Loads a dictionary
        :param dictionary: Dictionary to be loaded
        :param name_list: List of names to be updated
        """
        for name in dictionary:
            data = dictionary[name]
            if name_list is None or name in name_list:
                if isinstance(data, dict):
                    if name in self:
                        self[name].load(data)
                    else:
                        self[name] = Dict(data)
                elif isinstance(data, list):
                    self[name] = list()
                    for item in data:
                        if isinstance(item, dict):
                            self[name].append(Dict(item))
                        else:
                            self[name].append(item)
                else:
                    self[name] = data

    def save(self, path):
        """
        Saves the dictionary into a json file
        :param path: Path of the json file
        """
        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, 'cfg.json')

        with open(path, 'w') as file:
            json.dump(self, file, indent=True)


class Configuration(Dict):
    """
    Configuration loaded from a json file
    """
    def __init__(self, path: str, default_path=None):
        super(Configuration, self).__init__()

        if default_path is not None:
            self.load(default_path)

        self.load(path)

    def load_model(self, path: str):
        self.load(path, name_list=["model"])

    def load(self, path: str, name_list: list = None):
        """
        Loads attributes from a json file
        :param path: Path of the json file
        :param name_list: List of names to be updated
        :return:
        """
        with open(path) as file:
            data = json.loads(jsmin.jsmin(file.read()))

            super(Configuration, self).load(data, name_list)

