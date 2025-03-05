import os
import yaml
import json


class VerifierConfigs:
    def __init__(self, verifier):
        self.verifier = verifier
        self.load_default_configs()

    def load_default_configs(self):
        from .abcrown import ABCrown
        from .mnbab import MNBab

        dirname = os.path.dirname(__file__)
        if isinstance(self.verifier, ABCrown):
            default_configs_path = os.path.join(dirname, "abcrown/abcrown.yml")
            with open(default_configs_path, "r") as fp:
                self.configs = yaml.safe_load(fp)
        elif isinstance(self.verifier, MNBab):
            dirname = os.path.dirname(__file__)
            default_configs_path = os.path.join(dirname, "mnbab/mnbab.json")
            with open(default_configs_path, "r") as fp:
                self.configs = json.load(fp)
        else:
            raise NotImplementedError

    def save_configs(self, path):
        from .abcrown import ABCrown
        from .mnbab import MNBab

        if isinstance(self.verifier, ABCrown):
            data = yaml.dump(self.configs)
        elif isinstance(self.verifier, MNBab):
            data = json.dumps(self.configs, indent=4)
        else:
            raise NotImplementedError

        with open(path, "w") as fp:
            fp.write(data)
