from ..verifiers.abcrown import ABCrown
from ..verifiers.mnbab import MNBab
from ..verifiers.verinet import Verinet
from ..verifiers.nnenum import NNEnum
from ..verifiers.neuralsat import NeuralSat
from ..verifiers.veristable import VeriStable
from ..verifiers.marabou import Marabou

from .property import Property, LocalRobustnessProperty

import subprocess
import os


class VerificationProblem:
    def __init__(
        self,
        logger,
        property_configs,
        verifier,
        verifier_config,
        paths,
        greybox=False,
    ):
        self.logger = logger
        self.property_configs = property_configs
        self.paths = paths
        self.verifier_config = verifier_config
        self.init_verifiers(verifier)

    def init_verifiers(self, verifier):
        match verifier:
            case "abcrown22":
                configs = {
                    'version': 22,
                    'beta': True,
                }
                v = ABCrown(self, configs)
            case "abcrown23":
                configs = {
                    'version': 23,
                    'beta': True,
                }
                v = ABCrown(self, configs)
            case "abcrown24":
                configs = {
                    'version': 24,
                    'beta': True,
                }
                v = ABCrown(self, configs)
            case "abcrown25":
                configs = {
                    'version': 25,
                    'beta': True,
                }
                v = ABCrown(self, configs)
            case "mnbab":
                v = MNBab(self)
            case "verinet":
                v = Verinet(self)
            case "nnenum":
                v = NNEnum(self)
            case 'neuralsat':
                v = NeuralSat(self, version=1)
            case 'neuralsatp':
                v = NeuralSat(self, version=2)
            case 'neuralsatpp':
                v = NeuralSat(self, version=3)
            case 'veristable':
                v = VeriStable(self)
            case 'marabou':
                v = Marabou(self)
            case _:
                raise NotImplementedError(verifier)
        self.verifier = v

    def set_generic_property(self, path):
        self.logger.info(f"Using predefined generic property.")
        self.property = Property(self.logger)
        self.property.set(path)

    def generate_property(self, format="vnnlib", model_path=None):
        self.logger.info(f"Generating property ... ")
        
        if type(self.verifier) in [ABCrown, MNBab, Verinet, NNEnum, NeuralSat, VeriStable, Marabou]:
            assert self.property_configs["type"] == "local robustness"
            self.property = LocalRobustnessProperty(self.logger, self.property_configs)
            self.property.generate(self.paths["prop_dir"], format=format, model_path=model_path)
        else:
            raise NotImplementedError()
        self.logger.info(f"Property generated.")

    def harden_benchmark(self):

        model_path = self.paths["model_path"]
        output_path = model_path.split(".onnx")[0] + f"_harder_{self.property_configs['id']}.onnx"
        self.paths['harder_model_path'] = output_path

        # Check if the harden model is created
        if os.path.exists(output_path):
            self.logger.info(f"Skip hardening the model: {output_path}")
            return

        cmd = f"$SwarmHost/scripts/run_relusplitter.sh"
        cmd += f" --net {model_path}"
        cmd += f" --spec {self.property.property_path}"
        cmd += f" --output {output_path}"

        subprocess.run(cmd, shell=True)
        if not os.path.exists(output_path):
            self.logger.info(f"exceeds maximum protobuf size of 2GB")


    def verify(self):
        config_path = self.paths["veri_config_path"]
        property_path = self.property.property_path
        log_path = self.paths["veri_log_path"]
        time = self.verifier_config["time"]
        harder_model = self.paths["harder_model_path"]
        self.verifier.configure(config_path)
        
        self.logger.info("Verifying started")
        return self.verifier.run(config_path, harder_model, property_path, log_path, time)

    def analyze(self):
        return self.verifier.analyze()
