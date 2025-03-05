import os

from .. import Verifier
from ..verifier_configs import VerifierConfigs


class NeuralSat(Verifier):
    def __init__(self, verification_problem):
        super().__init__(verification_problem)
        self.__name__ = "NeuralSat"

    def configure(self, config_path):
        ...

    def run(self, config_path, model_path, property_path, log_path, time, memory):
        dataset = self.verification_problem.property.property_configs['artifact'].lower()
        
        cmd = f"$SwarmHost/scripts/run_neuralsat.sh --net {model_path} --spec {property_path} --dataset {dataset}"
        
        self.execute(cmd, log_path, time, memory)

    def analyze(self):
        with open(self.verification_problem.paths["veri_log_path"], "r") as fp:
            lines = fp.readlines()
        veri_ans, veri_time = super().pre_analyze(lines)

        if not (veri_ans and veri_time):
            veri_ans = None
            veri_time = None
            for l in lines[-100:]:
                if " UNSAT " in l:
                    veri_ans = 'unsat'
                elif " SAT " in l and 'Number of SAT variables' not in l:
                    veri_ans = 'sat'
                elif " UNKNOWN " in l:
                    veri_ans = 'unknown'
                if veri_ans:
                    veri_time = float(l.strip().split()[-1])
                    print(veri_time)
                    break

        assert (
            veri_ans and veri_time
        ), f"Answer: {veri_ans}, time: {veri_time}, log: {self.verification_problem.paths['veri_log_path']}"
        
        return super().post_analyze(veri_ans, veri_time)
