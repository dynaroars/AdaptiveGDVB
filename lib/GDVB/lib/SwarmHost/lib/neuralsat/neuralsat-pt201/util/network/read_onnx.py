from beartype import beartype
import onnxruntime as ort
import torch.nn as nn
import onnx2pytorch
import numpy as np
import collections
# import onnx2torch
import traceback
import warnings
import torch
import onnx
import gzip

from util.misc.error import *


custom_quirks = {
    'Reshape': {
        'fix_batch_size': False
    },
    'Transpose': {
        'merge_batch_size_with_channel': True,
        'remove_gdvb_transpose': True,
    },
    'Softmax' :{
        'skip_last_layer': True
    },
    'Squeeze' :{
        'skip_last_layer': True
    },
    'Conv' :{
        'merge_batch_norm': True
    },
}

@beartype
# def inference_onnx(path: str, *inputs: np.ndarray) -> list[np.ndarray]:
def inference_onnx(path: str, *inputs: np.ndarray):
    sess = ort.InferenceSession(onnx.load(path).SerializeToString())
    names = [i.name for i in sess.get_inputs()]
    return sess.run(None, dict(zip(names, inputs)))


@beartype
def add_batch(shape: tuple) -> tuple:
    if len(shape) == 1:
        return (1, shape[0])
    
    if shape[0] not in [-1, 1]:
        return (1, *shape)
    
    return shape
        

@beartype
def _parse_onnx(path: str) -> tuple:
    # print('Loading ONNX with customized quirks:', custom_quirks)
    onnx_model = onnx.load(path)
    
    onnx_inputs = [node.name for node in onnx_model.graph.input]
    initializers = [node.name for node in onnx_model.graph.initializer]
    inputs = list(set(onnx_inputs) - set(initializers))
    inputs = [node for node in onnx_model.graph.input if node.name in inputs]
    
    onnx_input_dims = inputs[0].type.tensor_type.shape.dim
    onnx_output_dims = onnx_model.graph.output[0].type.tensor_type.shape.dim
    
    orig_input_shape = tuple(d.dim_value if d.dim_value > 0 else 1 for d in onnx_input_dims)
    orig_output_shape = tuple(d.dim_value if d.dim_value > 0 else 1 for d in onnx_output_dims) if len(onnx_output_dims) else (1,)
    
    batched_input_shape = add_batch(orig_input_shape)
    batched_output_shape = add_batch(orig_output_shape)

    pytorch_model = onnx2pytorch.ConvertModel(onnx_model, experimental=True, quirks=custom_quirks)
    pytorch_model.eval()
    
    pytorch_model.to(torch.get_default_dtype())
    
    is_nhwc = pytorch_model.is_nhwc
    
    if custom_quirks.get('Softmax', {}).get('skip_last_layer', False):
        custom_quirks['Softmax']['skip_last_layer'] = pytorch_model.is_last_removed.get('Softmax', False)
    
    if custom_quirks.get('Squeeze', {}).get('skip_last_layer', False):
        custom_quirks['Squeeze']['skip_last_layer'] = pytorch_model.is_last_removed.get('Squeeze', False)
    
    # print('nhwc:', is_nhwc, batched_input_shape)
    
    # check conversion
    correct_conversion = True
    try:
        batch = 2
        dummy = torch.randn(batch, *batched_input_shape[1:], dtype=torch.get_default_dtype())
        # print(dummy.shape)
        output_onnx = torch.cat([torch.from_numpy(inference_onnx(path, dummy[i].view(orig_input_shape).float().numpy())[0]).view(batched_output_shape) for i in range(batch)])
        # print('output_onnx:', output_onnx)
        output_pytorch = pytorch_model(dummy.permute(0, 3, 1, 2) if is_nhwc else dummy).detach().numpy()
        # print('output_pytorch:', output_pytorch)
        correct_conversion = np.allclose(output_pytorch, output_onnx, 1e-5, 1e-5)
    except:
        raise OnnxConversionError

    if not correct_conversion and custom_quirks.get('Conv', {}).get('merge_batch_norm', False):
        raise OnnxMergeBatchNormError
    
    if not correct_conversion and not custom_quirks.get('Softmax', {}).get('skip_last_layer', False):
        raise OnnxOutputAllCloseError
    # else:
    #     print(pytorch_model)
    #     print(batched_input_shape)
    #     print(batched_output_shape)
    #     print('DEBUG: correct')
    #     exit()
        
    if is_nhwc:
        assert len(batched_input_shape) == 4
        n_, h_, w_, c_ = batched_input_shape
        batched_input_shape = (n_, c_, h_, w_)
    
    return pytorch_model, batched_input_shape, batched_output_shape, is_nhwc



@beartype
def parse_onnx(path: str) -> tuple:
    while True:
        try:
            return _parse_onnx(path)
        except OnnxMergeBatchNormError:
            custom_quirks['Conv']['merge_batch_norm'] = False
            continue
        except OnnxOutputAllCloseError:
            # print(f'[{i}] Model was converted incorrectly. Try again.')
            continue
        except OnnxConversionError:
            if not custom_quirks['Reshape']['fix_batch_size']:
                custom_quirks['Reshape']['fix_batch_size'] = True
                continue
            else:
                warnings.warn(f'Unable to convert onnx to pytorch model')
                traceback.print_exc()
                exit()
        except SystemExit:
            exit()
        except:
            warnings.warn(f'Unable to convert onnx to pytorch model')
            traceback.print_exc()
            exit()
            
            
            

def parse_pth(pth_path: str) -> tuple:
    pytorch_model = torch.load(pth_path)
    
    input_shape = (1, 3, 32, 32)
    output_shape = tuple(pytorch_model(torch.zeros(input_shape)).shape)
    is_nhwc = False
    
    custom_quirks['Softmax']['skip_last_layer'] = False
    custom_quirks['Squeeze']['skip_last_layer'] = False
    # check conversion
    # correct_conversion = True
    # assert correct_conversion

    return pytorch_model, input_shape, output_shape, is_nhwc