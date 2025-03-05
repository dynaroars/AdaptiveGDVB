# from ablation study of Hai

import os
import time
import argparse
import torch

import onnxruntime as ort
import onnx2pytorch
import numpy as np
import warnings
import onnx
import torch.nn as nn


from src.abstract_layers.abstract_network import AbstractNetwork
from src.mn_bab_verifier import MNBaBVerifier
from src.utilities.argument_parsing import get_args, get_config_from_json
from src.utilities.initialization import seed_everthing
from src.utilities.loading.network import freeze_network, load_net  # , load_onnx_net
from src.utilities.vnncomp_input_parsing import (
    parse_vnn_lib_prop,
    translate_box_to_sample,
    translate_constraints_to_label,
)

#  babsr, filtered_smart_branching, active_constraint_score
# simplify_cmd = (
#    "/home/droars/anaconda3/envs/dnnv/bin/python3 simplify_onnx.py {onnx_path}"
# )


def load_onnx_net(path):
    onnx_model = onnx.load(path)

    onnx_input_dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
    onnx_output_dims = onnx_model.graph.output[0].type.tensor_type.shape.dim
    orig_input_shape = tuple(d.dim_value for d in onnx_input_dims)
    batched_input_shape = (
        tuple(d.dim_value for d in onnx_input_dims)
        if len(onnx_input_dims) > 1
        else (1, onnx_input_dims[0].dim_value)
    )
    output_shape = (
        tuple(d.dim_value for d in onnx_output_dims)
        if len(onnx_output_dims) > 1
        else (1, onnx_output_dims[0].dim_value)
    )

    # convert ONNX to Pytorch model (experimental=True for supporting batch processing)
    pytorch_model = onnx2pytorch.ConvertModel(onnx_model, experimental=False)
    pytorch_model.eval()

    layers = list(pytorch_model.modules())[1:]
    sequence = []
    ignored = False
    for idx, layer in enumerate(layers):
        if isinstance(layer, nn.Linear) and not ignored:
            sequence.append(nn.Flatten())
            ignored = True

        if not ignored:
            continue

        if isinstance(layer, torch.nn.Linear):
            out_f, in_f = layer.weight.data.size()
            # print(in_f, out_f)
            new_layer = nn.Linear(in_f, out_f)
            # print(new_layer)
            # print(new_layer.weight.data.shape)
            # print(layer.weight.data.shape)
            # print()
            new_layer.weight.data.copy_(layer.weight.data)
            new_layer.bias.data.copy_(layer.bias.data)
            sequence.append(new_layer)
        else:
            sequence.append(layer)

        # print(idx, layer, type(layer))

    pytorch_model = nn.Sequential(*sequence)
    freeze_network(pytorch_model)

    print(pytorch_model)
    # check conversion
    correct_conversion = True
    try:
        dummy = torch.randn(batched_input_shape)
        # print(dummy.shape)
        output_pytorch = pytorch_model(dummy).detach().numpy()

        def inference_onnx(path, *inputs):
            sess = ort.InferenceSession(onnx.load(path).SerializeToString())
            names = [i.name for i in sess.get_inputs()]
            inp = dict(zip(names, inputs))
            res = sess.run(None, inp)
            return res

        output_onnx = inference_onnx(path, dummy.view(orig_input_shape).numpy())[0]
        # print(output_pytorch)
        # print(output_onnx)
        correct_conversion = np.allclose(output_pytorch, output_onnx, 1e-4, 1e-5)
    except:
        warnings.warn(f"Unable to check conversion correctness")
        import traceback

        print(traceback.format_exc())
        exit()

    if not correct_conversion:
        warnings.warn("Model was converted incorrectly.")
        exit()
    # else:
    #     print('DEBUG: correct')
    #     exit()

    return pytorch_model, batched_input_shape, output_shape


if __name__ == "__main__":
    START = time.time()
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", required=True)
    argparser.add_argument("--onnx_path", required=True)
    argparser.add_argument("--vnnlib_path", required=True)
    argparser.add_argument("--timeout", required=True)
    args = argparser.parse_args()

    config = get_config_from_json(args.config)
    seed_everthing(config.random_seed)

    if torch.cuda.is_available() and config.use_gpu:
        device = torch.device("cuda")
        # if config['bab_batch_size'][0] > 1:
        #     config['bab_batch_size'] = [1] * 6
    else:
        device = torch.device("cpu")

    # device = torch.device("cpu")
    # we work with pytorch nets, not onnx format
    netname = args.onnx_path

    # netname = netname.replace(".onnx", ".pyt")

    # if 'cifar' in os.path.basename(netname):
    #     config.input_dim = [3, 32, 32]
    # elif 'mnist' in os.path.basename(netname):
    #     config.input_dim = [1, 28, 28]
    # else:
    #     raise NotImplementedError

    vnn_lib_spec = args.vnnlib_path
    timeout_for_property = float(args.timeout)

    print("Configuration:", args.config)
    print("Network      :", netname)
    print("Spec         :", vnn_lib_spec)
    # n_neurons_per_layer, n_layers = None, None

    # try:
    #     network_size_info = netname.split("mnist-net_")[1].split(".pyt")[0]
    #     n_neurons_per_layer, n_layers = [int(s) for s in network_size_info.split("x")]
    # except IndexError:
    #     pass
    # original_network = load_net(netname, n_layers, n_neurons_per_layer)

    # os.system(simplify_cmd.format(onnx_path=netname))
    # netname = f'simplified.onnx'

    original_network, input_dim, output_dim = load_onnx_net(netname)
    print("input_dim     :", input_dim)
    config.input_dim = input_dim[1:]
    original_network.to(device)
    # exit()

    network = AbstractNetwork.from_concrete_module(original_network, config.input_dim)
    freeze_network(network)

    verifier = MNBaBVerifier(
        network,
        device,
        config.optimize_alpha,
        config.alpha_lr,
        config.alpha_opt_iterations,
        config.optimize_prima,
        config.prima_lr,
        config.prima_opt_iterations,
        config.prima_hyperparameters,
        config.peak_lr_scaling_factor,
        config.final_lr_div_factor,
        config.beta_lr,
        config.bab_batch_size,
        config.branching,
        config.recompute_intermediate_bounds_after_branching,
        config.use_dependence_sets,
        config.use_early_termination,
    )

    # exit()

    input_constraints, output_constraints = parse_vnn_lib_prop(vnn_lib_spec)
    input_lb_arr, input_ub_arr = input_constraints[0]
    input_lb = torch.tensor(input_lb_arr).view(config.input_dim).to(device)
    input_ub = torch.tensor(input_ub_arr).view(config.input_dim).to(device)

    label = translate_constraints_to_label(output_constraints)[0]

    original_images, __ = translate_box_to_sample(input_constraints, equal_limits=True)
    original_image = (
        torch.tensor(original_images[0]).view(config.input_dim).unsqueeze(0).to(device)
    )

    # pred_label = torch.argmax(original_network(original_image)).item()
    # if pred_label != label:
    #    print("Network fails on test image, skipping.")
    #    raise

    used_time = time.time() - START
    stat = verifier.verify(
        0, original_image, input_lb, input_ub, label, timeout_for_property - used_time
    )

    print(f"Result: {stat}")
    print(f"Time: {time.time() - START}")
