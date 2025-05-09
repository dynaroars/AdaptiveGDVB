import os
import sys
import random
import pickle
import copy
import time
import concurrent.futures
import threading
import torch

import numpy as np
from tqdm import tqdm

from ..artifacts.ACAS import ACAS
from ..artifacts.MNIST import MNIST
from ..artifacts.CIFAR10 import CIFAR10
from ..artifacts.DAVE2 import DAVE2
from ..artifacts.TaxiNet import TaxiNet
from ..nn.layers import Dense, Conv

from fractions import Fraction as F
from .verification_problem import VerificationProblem


class VerificationBenchmark:
    def __init__(self, name, dnn_configs, ca_configs, settings):
        self.name = name
        self.ca_configs = ca_configs
        self.settings = settings
        self.artifact = self._create_artifact(dnn_configs)
        self.settings.logger.info("Computing Factors")
        (self.parameters, self.fc_ids, self.conv_ids) = self._gen_parameters(ca_configs)
        self._debug_layer()
        self.settings.logger.info("Computing Covering Array")
        self.ca = self._gen_ca(ca_configs)
        self.settings.logger.info("Computing DNN Specifications")
        self.verification_problems = self._gen_network_specifications()

        self.training_loss = {}
        self.verification_results = {}

    def _create_artifact(self, dnn_configs):
        if dnn_configs["artifact"] in globals():
            artifact = globals()[dnn_configs["artifact"]](dnn_configs)
        else:
            raise NotImplementedError(
                f'Unimplemented artifact: {dnn_configs["artifact"]}'
            )
        return artifact

    def _gen_parameters(self, ca_configs):
        # calculate fc/conv ids
        fc_ids = []
        conv_ids = []
        for i, x in enumerate(self.artifact.layers):
            if isinstance(x, Dense):
                fc_ids += [i]
            elif isinstance(x, Conv):
                conv_ids += [i]
            else:
                pass
        fc_ids = fc_ids[:-1]  # remove output layer
        parameters = {}
        # parse the parameters in the expressive way
        if "explicit" in ca_configs["parameters"]:
            for x in ca_configs["parameters"]["explicit"]:
                parameters[x] = ca_configs["parameters"]["explicit"][x]
        # parse the parameters described in number of levels and ranges
        else:
            for key in ca_configs["parameters"]["level"]:
                level_min = F(ca_configs["parameters"]["range"][key][0])
                level_max = F(ca_configs["parameters"]["range"][key][1])
                level_size = F(ca_configs["parameters"]["level"][key])

                if level_size == 1:
                    assert level_min == level_max
                    level = np.array([level_min], dtype=np.float32)
                else:
                    level_range = level_max - level_min
                    level_step = level_range / (level_size - 1)
                    level = np.arange(level_min, level_max + level_step, level_step)
                    level = np.array(level, dtype=np.float32)
                if key == "prop":
                    level = np.array(level, dtype=np.int32)

                assert len(level) == level_size
                parameters[key] = level
                # make sure all parameters are passed
                assert len(parameters[key]) == ca_configs["parameters"]["level"][key]
        return parameters, fc_ids, conv_ids

    def _debug_layer(self):
        # debug remaining layers
        if "fc" in self.parameters:
            prm_str = "Possible remaining # of FC layers: "
            rml = sorted(
                [
                    str(int(round(len(self.fc_ids) * (self.parameters["fc"][i]))))
                    for i in range(len(self.parameters["fc"]))
                ]
            )
            prm_str += " ".join(rml)
            self.settings.logger.debug(prm_str)

        if "conv" in self.parameters:
            prm_str = "Possible remaining # of Conv layers: "
            rml = sorted(
                [
                    str(int(round(len(self.conv_ids) * (self.parameters["conv"][i]))))
                    for i in range(len(self.parameters["conv"]))
                ]
            )
            prm_str += " ".join(rml)
            self.settings.logger.debug(prm_str)

        # print factor and levels
        self.settings.logger.debug("Factor and levels:")
        for key in self.parameters:
            self.settings.logger.debug(f"{key}: {self.parameters[key]}")

    def _gen_ca(self, ca_configs):
        # compute the covering array
        lines = ["[System]", f"Name: {self.name}", "", "[Parameter]"]

        for key in self.parameters:
            bits = [str(x) for x in range(len(self.parameters[key]))]
            bits = ",".join(bits)
            lines += [f"{key}(int) : {bits}"]

        if "constraints" in ca_configs:
            lines += ["[Constraint]"]
            constraints = ca_configs["constraints"]["value"]
            for con in constraints:
                lines += [con]

        lines = [x + "\n" for x in lines]

        strength = ca_configs["strength"]

        ca_config_path = os.path.join(
            self.settings.root, "cas", f"ca_config_{hash(self.settings)}.txt"
        )
        ca_path = os.path.join(
            self.settings.root, "cas", f"ca_{hash(self.settings)}.txt"
        )

        open(ca_config_path, "w").writelines(lines)

        acts_path = os.environ["acts_path"]
        if not os.path.exists(acts_path):
            raise FileNotFoundError(f"CA generator ACTS is not found at :{acts_path}")
        cmd = f"java  -Ddoi={strength} -jar {acts_path} {ca_config_path} {ca_path} > /dev/null"
        os.system(cmd)

        lines = open(ca_path, "r").readlines()

        vp_configs = []
        i = 0
        while i < len(lines):
            l = lines[i]
            if "Number of configurations" in l:
                nb_tests = int(l.strip().split(" ")[-1])
                self.settings.logger.info(f"# problems: {nb_tests}")

            if "Configuration #" in l:
                vp = []
                for j in range(len(self.parameters)):
                    l = lines[j + i + 2]
                    vp += [int(l.strip().split("=")[-1])]
                assert len(vp) == len(self.parameters)
                vp_configs += [vp]
                i += j + 2
            i += 1
        assert len(vp_configs) == nb_tests
        vp_configs = sorted(vp_configs)

        vp_configs_ = []
        for vpc in vp_configs:
            assert len(vpc) == len(self.parameters)
            tmp = {}
            for i, key in enumerate(self.parameters):
                tmp[key] = self.parameters[key][vpc[i]]
            vp_configs_ += [tmp]
        vp_configs = vp_configs_

        return vp_configs

    def _gen_network_specifications(self):
        network_specifications = []
        for vpc in self.ca:
            self.settings.logger.debug(f"Configuring verification problem: {vpc}")
            self.settings.logger.debug("----------Original Network----------")
            self.settings.logger.debug(
                f"Number neurons: {np.sum(self.artifact.onnx.nb_neurons)}"
            )
            for i, x in enumerate(self.artifact.layers):
                self.settings.logger.debug(f"{i}: {x}")

            # factor: neu
            if "neu" in vpc:
                neuron_scale_factor = vpc["neu"]
            else:
                neuron_scale_factor = 1

            # factor: fc, conv
            drop_ids = []
            add_ids = []
            layers_add = []
            if "fc" in vpc:
                if vpc["fc"] < 1:
                    # randomly select layers to drop
                    if self.settings.training_configs["drop_scheme"] == "random":
                        drop_fc_ids = sorted(
                            random.sample(
                                self.fc_ids,
                                int(round(len(self.fc_ids) * (1 - vpc["fc"]))),
                            )
                        )
                        drop_ids += drop_fc_ids
                    else:
                        raise NotImplementedError(
                            f"Unsupported drop scheme: "
                            f'{self.settings.training_configs["drop_scheme"]}'
                        )
                elif vpc["fc"] > 1:
                    # append fully connected layers to the last hidden layer
                    # nb_neurons = nb_neurons in the last hidden layer
                    if self.settings.training_configs["add_scheme"] == "last_same_relu":
                        last_layer_id = np.max(self.fc_ids)
                        nb_layer_to_add = np.arange(
                            1, round(len(self.fc_ids) * (vpc["fc"] - 1)) + 1, 1
                        )
                        fc_add_ids = [x + last_layer_id for x in nb_layer_to_add]
                        nb_neurons = self.artifact.layers[last_layer_id].size
                        layer = {
                            "layer_type": "FullyConnected",
                            "parameters": nb_neurons,
                            "activation_function": "relu",
                            "layer_id": fc_add_ids,
                        }
                        layers_add += [layer]
                        add_ids += fc_add_ids
                    else:
                        raise NotImplementedError(
                            f"Unsupported drop scheme: "
                            f'{self.settings.training_configs["drop_scheme"]}'
                        )
                else:
                    pass

            if "conv" in vpc:
                if vpc["conv"] < 1:
                    # randomly select layers to drop
                    if self.settings.training_configs["drop_scheme"] == "random":
                        drop_conv_ids = sorted(
                            random.sample(
                                self.conv_ids,
                                int(round(len(self.conv_ids) * (1 - vpc["conv"]))),
                            )
                        )
                        drop_ids += drop_conv_ids

                    else:
                        raise NotImplementedError(
                            f"Unsupported add scheme: "
                            f'{self.settings.training_configs["add_scheme"]}'
                        )
                elif vpc["conv"] > 1:
                    # append convolutional layers to the last hidden layer
                    # nb_neurons = nb_neurons in the last hidden layer
                    if self.settings.training_configs["add_scheme"] == "last_same_relu":
                        last_layer_id = np.max(self.conv_ids)
                        nb_layer_to_add = np.arange(
                            1, round(len(self.conv_ids) * (vpc["conv"] - 1)) + 1, 1
                        )
                        conv_add_ids = [x + last_layer_id for x in nb_layer_to_add]
                        last_layer = self.artifact.layers[last_layer_id]
                        assert isinstance(last_layer, Conv)
                        nb_kernels = last_layer.size
                        kernel_size = last_layer.kernel_size
                        stride = last_layer.stride
                        padding = last_layer.padding
                        layer = {
                            "layer_type": "Convolutional",
                            ## TODO: only supports SAME padding in order to maintain output shape
                            # "parameters": [nb_kernels, kernel_size, stride, padding],
                            # padding: VALID: 0, SAME: -1
                            "parameters": [nb_kernels, kernel_size, 1, -1],
                            "activation_function": "relu",
                            "layer_id": conv_add_ids,
                        }
                        layers_add += [layer]
                        add_ids += conv_add_ids
                    else:
                        raise NotImplementedError(
                            f"Unsupported add scheme: "
                            f'{self.settings.training_configs["add_scheme"]}'
                        )
                else:
                    pass

            if "lay" in vpc:
                if vpc["lay"] < 1:
                    # randomly select layers to drop
                    if self.settings.training_configs["drop_scheme"] == "random":
                        drop_fc_ids = sorted(
                            random.sample(
                                self.fc_ids,
                                int(round(len(self.fc_ids) * (1 - vpc["fc"]))),
                            )
                        )
                        drop_ids += drop_fc_ids
                    else:
                        raise NotImplementedError(
                            f"Unsupported drop scheme: "
                            f'{self.settings.training_configs["drop_scheme"]}'
                        )
                elif vpc["lay"] > 1:
                    # append fully connected layers to the last hidden layer
                    # nb_neurons = nb_neurons in the last hidden layer
                    if self.settings.training_configs["add_scheme"] == "last_same_relu":
                        last_layer_id = np.max(self.fc_ids)
                        nb_layer_to_add = np.arange(
                            1, round(len(self.fc_ids) * (vpc["fc"] - 1)) + 1, 1
                        )
                        fc_add_ids = [x + last_layer_id for x in nb_layer_to_add]
                        nb_neurons = self.artifact.layers[last_layer_id].size
                        layer = {
                            "layer_type": "FullyConnected",
                            "parameters": nb_neurons,
                            "activation_function": "relu",
                            "layer_id": fc_add_ids,
                        }
                        layers_add += [layer]
                        add_ids += fc_add_ids
                    else:
                        raise NotImplementedError(
                            f"Unsupported drop scheme: "
                            f'{self.settings.training_configs["drop_scheme"]}'
                        )
                else:
                    pass
            n = VerificationProblem(self.settings, vpc, self)
            dis_strats = [["drop", x] for x in drop_ids]
            dis_strats += [["add", x] for x in layers_add]

            # calculate data transformations, input dimensions

            if "transform" in n.distillation_config["distillation"]["data"]:
                transform = n.distillation_config["distillation"]["data"]["transform"][
                    "student"
                ]
                new_height = transform["height"]
            else:
                transform = None

            # input dimensions
            if "idm" in vpc:
                id_f = vpc["idm"]
                height = transform["height"]
                width = transform["width"]
                new_height = int(np.sqrt(id_f) * height)
                new_width = int(np.sqrt(id_f) * width)
                transform["height"] = new_height
                transform["width"] = new_width

                if new_height != height:
                    assert new_width != width
                    dis_strats += [["scale_input", np.sqrt(id_f)]]

            # input domain size
            if "ids" in vpc:
                mean = transform["mean"]
                max_value = transform["max_value"]
                min_value = transform["min_value"]
                ids_f = vpc["ids"]

                transform["mean"] = [float(x * ids_f) for x in mean]
                transform["max_value"] = float(max_value * ids_f)
                transform["min_value"] = float(min_value * ids_f)

            if self.artifact.onnx.input_format == "NCHW":
                nb_channel = self.artifact.onnx.input_shape[0]
                input_shape = [nb_channel, new_height, new_height]
            elif self.artifact.onnx.input_format == "NHWC":
                nb_channel = self.artifact.onnx.input_shape[2]
                input_shape = [nb_channel, new_height, new_height]
            elif self.artifact.onnx.input_format == "ACAS":
                input_shape = self.artifact.onnx.input_shape
            else:
                raise ValueError(
                    f"Unrecognized ONNX input format: {self.artifact.onnx.input_format}"
                )

            # set up new network with added and dropped layers
            n.set_distillation_strategies(dis_strats)
            n.calc_order("nb_neurons", self.artifact.layers, input_shape)

            # calculate real scale factors
            if "neu" in vpc:
                neuron_scale_factor = (
                    np.sum(self.artifact.onnx.nb_neurons[:-1]) * neuron_scale_factor
                ) / np.sum(n.nb_neurons[:-1])
            elif "fc" in vpc or "conv" in vpc:
                neuron_scale_factor = np.sum(
                    self.artifact.onnx.nb_neurons[:-1]
                ) / np.sum(n.nb_neurons[:-1])

            # assign scale factors
            if neuron_scale_factor != 1:
                # calculate scale ids
                if "neu" in vpc or "fc" in vpc or "conv" in vpc:
                    scale_ids = set(self.fc_ids + self.conv_ids)
                    scale_ids = set(scale_ids) - set(drop_ids)
                    scale_ids = list(scale_ids)
                    # print(scale_ids)

                    if add_ids:
                        for add_id in add_ids:
                            for i in range(len(scale_ids)):
                                if scale_ids[i] >= add_id:
                                    scale_ids[i] += 1
                            scale_ids = sorted(scale_ids + [add_id])
                            # print(scale_ids)
                    # print(scale_ids)
                    # print()
                else:
                    scale_ids = []
                self.settings.logger.debug("Computing layer scale factors ...")
                self.settings.logger.debug(
                    f"Layers to Add: {add_ids}, Delete: {drop_ids}."
                )
                self.settings.logger.debug(f"Layers to Scale: {scale_ids}.")
                # print(n.fc_and_conv_kernel_sizes, scale_ids)
                for x in scale_ids:
                    assert n.fc_and_conv_kernel_sizes[x] > 0
                    if int(n.fc_and_conv_kernel_sizes[x] * neuron_scale_factor) == 0:
                        # self.settings.logger.warn(
                        #    "Detected small layer scale factor, layer size is rounded up to 1."
                        # )
                        dis_strats += [["scale", x, 1 / n.fc_and_conv_kernel_sizes[x]]]
                    else:
                        dis_strats += [["scale", x, neuron_scale_factor]]
                n = VerificationProblem(self.settings, vpc, self)
                n.set_distillation_strategies(dis_strats)
                n.calc_order("nb_neurons", self.artifact.layers, input_shape)

            if transform:
                n.distillation_config["distillation"]["data"]["transform"][
                    "student"
                ] = transform

            n.set_misc_parameters()
            network_specifications += [n]
            # self.settings.logger.debug("----------New Network----------")
            # self.settings.logger.debug(f"Number neurons: {np.sum(n.nb_neurons)}")
            # for i, x in enumerate(n.layers):
            #    self.settings.logger.debug(f"{i}: {x}")

        self.settings.logger.info(f"# NN: {len(network_specifications)}")
        return network_specifications

    def train(self):
        self.settings.logger.info("Training ...")
        # filter repeated network specifications
        nets_to_train = {x.net_name: x for x in self.verification_problems}
        nets_to_train = [nets_to_train[x] for x in nets_to_train]

        progress_bar = tqdm(
            total=len(nets_to_train), desc="Training ... ", ascii=False, file=sys.stdout
        )

        # def concurrent_train(network):
        #     self.settings.logger.info(f"Training network {network.net_name} with current thread: {threading.current_thread().getName()}")
        #     network.train()

        # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        #     future_to_train_network = {executor.submit(concurrent_train, n): n.net_name for n in nets_to_train}
        #     for future in concurrent.futures.as_completed(future_to_train_network):
        #         self.settings.logger.info(f"Finished network: {future_to_train_network[future]} ...")
        #         progress_bar.update(1)
        #         progress_bar.refresh()

        # Train a network
        for n in nets_to_train:
            self.settings.logger.info(f"Training network: {n.net_name} ...")
            # Clear CUDA memory
            with torch.no_grad():
                torch.cuda.empty_cache()

            n.train()
            progress_bar.update(1)
            progress_bar.refresh()

            # Clear CUDA memory
            with torch.no_grad():
                torch.cuda.empty_cache()

        progress_bar.close()

    def trained(self, count=False):
        trained = [x.trained(True) for x in self.verification_problems]
        if count:
            return trained.count(True)
        else:
            return all(trained)

    def gen_props(self):
        self.settings.logger.info("Generating properties ...")

        progress_bar = tqdm(
            total=len(self.verification_problems),
            desc="Generating ... ",
            ascii=False,
            file=sys.stdout,
        )
        for vp in self.verification_problems:
            vp.gen_prop()
            progress_bar.update(1)
            progress_bar.refresh()
        progress_bar.close()

    def critical_region_analysis(self):
        self.settings.logger.info("Critical Region Analysis ...")

        if "eps" in self.parameters:
            raise ValueError(
                "Critical region analysis cannot be performed with customized eps regions."
            )

        verification_problem_backup = self.verification_problems

        def calculate_res():
            all_res = []
            for n in self.verification_results:
                for v in self.verification_results[n]:
                    all_res += [
                        self.settings.answer_code[self.verification_results[n][v][0]]
                    ]
            unique, counts = np.unique(np.array(all_res), return_counts=True)
            res = dict(zip(unique, counts))
            for i in range(1, len(list(self.settings.answer_code.keys())) + 1):
                if i not in res:
                    res[i] = 0
            return res

        iteration = 0
        # 1)  initial settings
        self.settings.logger.debug(f"CRA binary search iteration: {iteration}")
        cravp = verification_problem_backup
        new_cravp = []
        for i, x in enumerate(cravp):
            assert "eps" not in x.vpc
            y = copy.deepcopy(x)
            y.gen_names()
            y.settings.verification_configs["time"] = 10
            new_cravp += [y]
        self.verification_problems = new_cravp

        print("CRA: verifying ... ")
        # 2. verify the new benchmark
        self.verify()

        # 3. wait for verification
        nb_verification_tasks = len(self.verification_problems)
        progress_bar = tqdm(
            total=nb_verification_tasks,
            desc="Waiting on verification ... ",
            ascii=False,
            file=sys.stdout,
        )
        nb_verified_pre = self.verified(True)
        progress_bar.update(nb_verified_pre)
        while not self.verified():
            time.sleep(10)
            nb_verified_now = self.verified(True)
            progress_bar.update(nb_verified_now - nb_verified_pre)
            progress_bar.refresh()
            nb_verified_pre = nb_verified_now
        progress_bar.close()

        print("CRA: analying verification ... ")
        # 4. analyze verification results
        self.analyze_verification()

        total_problems = len(self.verification_problems)
        res = calculate_res()
        del self.verification_problems
        self.verification_results = {}

        def bi_search(res, now, low, high, kind, threshold=0.25):
            # 1. Create a new benchmark with scaling eps, limit time to be 10 seconds
            stop = False
            if kind == "lb":
                solved_ratio = res[1] / total_problems
                if abs(solved_ratio - threshold) < 0.05:
                    stop = True
                if solved_ratio < threshold:
                    high = now
                    now = (now - low) / 2
                else:
                    low = now
                    now = (high - now) / 2
            elif kind == "ub":
                solved_ratio = res[2] / total_problems
                if abs(solved_ratio - threshold) < 0.05:
                    stop = True
                if solved_ratio < threshold:
                    low = now
                    now = (high - now) / 2
                else:
                    high = now
                    now = (now - low) / 2
            else:
                assert False

            cravp = verification_problem_backup
            new_cravp = []
            for x in cravp:
                assert "eps" not in x.vpc
                y = copy.deepcopy(x)
                y.vpc["eps"] = now
                y.settings.verification_configs["time"] = 10
                y.gen_names()
                new_cravp += [y]

            self.verification_problems = new_cravp

            # 2. verify the new benchmark
            print("CRA: verifying ... ")
            self.verify()

            # 3. wait for verification
            nb_verification_tasks = len(self.verification_problems)
            progress_bar = tqdm(
                total=nb_verification_tasks,
                desc="Waiting on verification ... ",
                ascii=False,
                file=sys.stdout,
            )
            nb_verified_pre = self.verified(True)
            progress_bar.update(nb_verified_pre)
            while not self.verified():
                time.sleep(10)
                nb_verified_now = self.verified(True)
                progress_bar.update(nb_verified_now - nb_verified_pre)
                progress_bar.refresh()
                nb_verified_pre = nb_verified_now
            progress_bar.close()

            print("CRA: analying verification ... ")
            # 4. analyze verification results
            self.analyze_verification()

            res = calculate_res()
            del self.verification_problems
            self.verification_results = {}

            return res, now, low, high, stop

        res_lb = res
        now_lb = 1
        low_lb = 0
        high_lb = 1 / self.settings.verification_configs["eps"]
        stop_lb = False

        res_ub = res
        now_ub = 1
        low_ub = 0
        high_ub = 1 / self.settings.verification_configs["eps"]
        stop_ub = False

        print(f"[{iteration}]LB: ", res_lb, now_lb, low_lb, high_lb, stop_lb)
        print(f"[{iteration}]UB: ", res_ub, now_ub, low_ub, high_ub, stop_ub)

        while iteration < 10 or (stop_lb and stop_ub):
            iteration += 1
            self.settings.logger.debug(f"CRA binary search iteration: {iteration}")

            if not stop_lb:
                res_lb, now_lb, low_lb, high_lb, stop_lb = bi_search(
                    res_lb, now_lb, low_lb, high_lb, "lb"
                )
            if not stop_ub:
                res_ub, now_ub, low_ub, high_ub, stop_ub = bi_search(
                    res_ub, now_ub, low_ub, high_ub, "ub"
                )

            print(f"[{iteration}]LB: ", res_lb, now_lb, low_lb, high_lb, stop_lb)
            print(f"[{iteration}]UB: ", res_ub, now_ub, low_ub, high_ub, stop_ub)

        self.settings.logger.info("Critical Region Analysis done.")

        exit()

    def verify(self):
        self.settings.logger.info("Verifying ...")
        vp_tool_verifiers = []
        for vp in self.verification_problems:
            for tool in self.settings.verification_configs["verifiers"]:
                for options in self.settings.verification_configs["verifiers"][tool]:
                    vp_tool_verifiers += [(vp, tool, options)]

        progress_bar = tqdm(
            total=len(vp_tool_verifiers),
            desc="Verifying ... ",
            ascii=False,
            file=sys.stdout,
        )
        for vp_tv in vp_tool_verifiers:
            vp, tool, options = vp_tv
            self.settings.logger.info(
                f"Verifying {vp.vp_name} with {tool}:[{options}] ..."
            )
            if tool != "SwarmHost":
                vp.gen_prop()

            
            # Clear CUDA memory
            with torch.no_grad():
                torch.cuda.empty_cache()
            vp.verify(tool, options)
            # Clear CUDA memory
            with torch.no_grad():
                torch.cuda.empty_cache()
            progress_bar.update(1)
            progress_bar.refresh()
        progress_bar.close()

    def verified(self, count=False):
        verified = [x.verified() for x in self.verification_problems]
        if count:
            return verified.count(True)
        else:
            return all(verified)

    def analyze_all(self):
        self.analyze_training()
        self.analyze_verification()

    def analyze_training(self):
        for vp in self.verification_problems:
            self.training_loss[vp.vp_name] = vp.analyze_training()

    def analyze_verification(self):
        for vp in self.verification_problems:
            self.verification_results[vp.vp_name] = vp.analyze_verification()
            # print(f'rm {vp.veri_log_path}')

    def save_results(self):
        save_path_prefix = os.path.join(
            self.settings.root, f"{self.name}_{self.settings.seed}"
        )

        train_loss_lines = []
        # save training losses as csv
        for x in self.training_loss:
            line = f"{x},"
            for xx in self.training_loss[x]:
                line = line + f"{xx}, "
            train_loss_lines += [f"{line}\n"]

        with open(f"{save_path_prefix}_train_loss.csv", "w") as handle:
            handle.writelines(train_loss_lines)

        # transpose verification results
        # from: results[problems][verifiers]
        # to:   results[verifiers][problems]
        results = {}
        verifiers = [x for x in list(self.verification_results.values())[0]]
        for p in self.verification_results:
            for v in verifiers:
                if v not in results.keys():
                    results[v] = {}
                assert p not in results[v].keys()
                results[v][p] = self.verification_results[p][v]

        # save verification results as pickle
        with open(f"{save_path_prefix}_verification_results.pickle", "wb") as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # calculate scr and par2 scores

        scr_dict = {}
        par_2_dict = {}
        for v in verifiers:
            sums = []
            scr = 0
            for p in results[v]:
                verification_answer = results[v][p][0]
                verification_time = results[v][p][1]
                if verification_answer in ["sat", "unsat"]:
                    scr += 1
                    sums += [verification_time]
                elif verification_answer in [
                    "unknown",
                    "error",
                    "timeout",
                    "memout",
                    "exception",
                    "rerun",
                    "torun",
                    "unrun",
                    "undetermined",
                ]:
                    sums += [self.settings.verification_configs["time"] * 2]
                else:
                    assert False
            par_2 = np.mean(np.array(sums))

            if v not in scr_dict.keys():
                scr_dict[v] = [scr]
                par_2_dict[v] = [par_2]
            else:
                scr_dict[v] += [scr]
                par_2_dict[v] += [par_2]

        print("")
        print("|{:>15} | {:>15} | {:>15}|".format("Verifier", "SCR", "PAR-2"))
        print("|----------------|-----------------|----------------|")
        for v in verifiers:
            print(
                "|{:>15} | {:>15} | {:>15.2f}|".format(
                    v, scr_dict[v][0], round(float(par_2_dict[v][0]), 2)
                )
            )
