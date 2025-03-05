import time
import argparse
import onnx

from onnx import numpy_helper

from verinet.parsers.onnx_parser import ONNXParser
from verinet.parsers.vnnlib_parser import VNNLIBParser
from verinet.verification.verinet import VeriNet
from verinet.verification.verifier_util import Status


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Verinet executor with onnx model and vnnlib property"
    )
    parser.add_argument("onnx_model", type=str)
    parser.add_argument("vnnlib", type=str)
    parser.add_argument("timeout", type=int)
    parser.add_argument("--input_shape", type=int, nargs="+")
    parser.add_argument("--max_procs", type=int, default=1)
    parser.add_argument("--transpose_fc_weights", type=bool, default=False)
    parser.add_argument("--use_64bit", type=bool, default=False)
    parser.add_argument("--use_gpu", type=bool, default=False)
    return parser.parse_args()


def main(args):
    solver = VeriNet(max_procs=args.max_procs, use_gpu=args.use_gpu)

    om = onnx.load(args.onnx_model)
    onnx.checker.check_model(om)
    # shape = numpy_helper.to_array(om.graph.node[0].attribute[0].t)
    # print("Shape:", shape, args.input_shape)
    print(om.graph.node)

    onnx_parser = ONNXParser(
        args.onnx_model,
        transpose_fc_weights=args.transpose_fc_weights,
        use_64bit=args.use_64bit,
    )
    model = onnx_parser.to_pytorch()
    model.eval()

    print("XXXXXXXXXXXXXXXXXXXX: model done.")
    vnnlib_parser = VNNLIBParser(args.vnnlib)
    objectives = vnnlib_parser.get_objectives_from_vnnlib(
        model, input_shape=args.input_shape
    )
    print(objectives[0])
    print("XXXXXXXXXXXXXXXXXXXX: vnnlib done")
    start_time = time.time()
    branches = 0
    max_depth = 0
    print("nb objectives", len(objectives))

    for i, objective in enumerate(objectives):
        print(f"XXXXXXXXXXXXXXXXXXXX: objective {i}")
        status = solver.verify(objective, args.timeout - (time.time() - start_time))

        branches += solver.branches_explored
        max_depth = max(max_depth, solver.max_depth)

        if (time.time() - start_time) > args.timeout:
            status = Status.Undecided
            break

        if status != Status.Safe:
            break
        print(f"XXXXXXXXXXXXXXXXXXXX: objective {i} done")

    print("XXXXXXXXXXXXXXXXXXXX: objectives done")

    print(f"Branches explored: {branches}")
    print(f"max depth: {max_depth}")
    print(f"Result: {status}")
    print(f"Time: {time.time() - start_time}")


if __name__ == "__main__":
    main(_parse_args())
