#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
""" Normalization operators"""
import copy

import torch
import torch.nn as nn

from .base import *
from .constant import BoundConstant
from .leaf import BoundParams
from .solver_utils import grb


class BoundBatchNormalization(Bound):
    def __init__(self, attr, inputs, output_index, options, training):
        super().__init__(attr, inputs, output_index, options)
        self.eps = attr['epsilon']
        self.momentum = round(1 - attr['momentum'], 5)  # take care!
        self.options = options.get("bn", {})
        # modes:
        #   - forward: use mean and variance estimated from clean forward pass
        #   - ibp: use mean and variance estimated from ibp
        self.bn_mode = self.options.get("mode", "forward")
        self.use_mean = self.options.get("mean", True)
        self.use_var = self.options.get("var", True)
        self.use_affine = self.options.get("affine", True)
        self.training = training
        self.patches_start = True
        self.mode = options.get("conv_mode", "matrix")
        if not self.use_mean or not self.use_var:
            logger.info(f'Batch normalization node {self.name}: use_mean {self.use_mean}, use_var {self.use_var}')

    def _check_unused_mean_or_var(self):
        # Check if either mean or var is opted out
        if not self.use_mean:
            self.current_mean = torch.zeros_like(self.current_mean)
        if not self.use_var:
            self.current_var = torch.ones_like(self.current_var)

    def forward(self, x, w, b, m, v):
        if len(x.shape) == 2:
            self.patches_start = False
        if self.training:
            dim = [0] + list(range(2, x.ndim))
            self.current_mean = x.mean(dim)
            self.current_var = x.var(dim, unbiased=False)
        else:
            self.current_mean = m.data
            self.current_var = v.data
        self._check_unused_mean_or_var()
        if not self.use_affine:
            w = torch.ones_like(w)
            b = torch.zeros_like(b)
        result = F.batch_norm(x, m, v, w, b, self.training, self.momentum, self.eps)
        if not self.use_mean or not self.use_var:
            # If mean or variance is disabled, recompute the output from self.current_mean
            # and self.current_var instead of using standard F.batch_norm.
            w = w / torch.sqrt(self.current_var + self.eps)
            b = b - self.current_mean * w
            shape = (1, -1) + (1,) * (x.ndim - 2)
            result = w.view(*shape) * x + b.view(*shape)
        return result

    def bound_forward(self, dim_in, *x):
        inp = x[0]
        assert (x[1].lower == x[1].upper).all(), "unsupported forward bound with perturbed mean"
        assert (x[2].lower == x[2].upper).all(), "unsupported forward bound with perturbed var"
        weight, bias = x[1].lower, x[2].lower
        if not self.training:
            assert (x[3].lower == x[3].upper).all(), "unsupported forward bound with perturbed mean"
            assert (x[4].lower == x[4].upper).all(), "unsupported forward bound with perturbed var"
            self.current_mean = x[3].lower
            self.current_var = x[4].lower
        self._check_unused_mean_or_var()
        if not self.use_affine:
            weight = torch.ones_like(weight)
            bias = torch.zeros_like(bias)

        tmp_bias = bias - self.current_mean / torch.sqrt(self.current_var + self.eps) * weight
        tmp_weight = weight / torch.sqrt(self.current_var + self.eps)

        tmp_weight = tmp_weight.view(*((1, 1, -1) + (1,) * (inp.lw.ndim - 3)))
        new_lw = torch.clamp(tmp_weight, min=0.) * inp.lw + torch.clamp(tmp_weight, max=0.) * inp.uw
        new_uw = torch.clamp(tmp_weight, min=0.) * inp.uw + torch.clamp(tmp_weight, max=0.) * inp.lw

        tmp_weight = tmp_weight.view(*((1, -1) + (1,) * (inp.lb.ndim - 2)))
        tmp_bias = tmp_bias.view(*((1, -1) + (1,) * (inp.lb.ndim - 2)))
        new_lb = torch.clamp(tmp_weight, min=0.) * inp.lb + torch.clamp(tmp_weight, max=0.) * inp.ub + tmp_bias
        new_ub = torch.clamp(tmp_weight, min=0.) * inp.ub + torch.clamp(tmp_weight, max=0.) * inp.lb + tmp_bias

        return LinearBound(
            lw = new_lw,
            lb = new_lb,
            uw = new_uw,
            ub = new_ub)

    def bound_backward(self, last_lA, last_uA, *x, **kwargs):
        assert not self.is_input_perturbed(1) and not self.is_input_perturbed(2), \
            'Weight perturbation is not supported for BoundBatchNormalization'

        def get_param(p):
            if isinstance(p, BoundConstant):
                # When affine is disabled in BN
                return p.value
            elif isinstance(p, BoundParams):
                return p.param
            else:
                raise TypeError(p)

        # x[0]: input, x[1]: weight, x[2]: bias, x[3]: running_mean, x[4]: running_var
        weight = get_param(x[1])
        bias = get_param(x[2])
        if not self.training:
            self.current_mean = x[3].value
            self.current_var = x[4].value
        self._check_unused_mean_or_var()
        if not self.use_affine:
            weight = torch.ones_like(weight)
            bias = torch.zeros_like(bias)

        tmp_bias = bias - self.current_mean / torch.sqrt(self.current_var + self.eps) * weight
        tmp_weight = weight / torch.sqrt(self.current_var + self.eps)

        def _bound_oneside(last_A):
            if last_A is None:
                return None, 0
            if type(last_A) == Tensor:
                next_A = last_A * tmp_weight.view(*((1, 1, -1) + (1,) * (last_A.ndim - 3)))
                if last_A.ndim > 3:
                    sum_bias = (last_A.sum(tuple(range(3, last_A.ndim))) * tmp_bias).sum(2)
                else:
                    sum_bias = (last_A * tmp_bias).sum(2)
            elif type(last_A) == Patches:
                # TODO Only 4-dim BN supported in the Patches mode
                if last_A.identity == 0:
                    # FIXME (09/17): Need to check if it has already been padding.
                    # Patch has dimension (out_c, batch, out_h, out_w, c, h, w) or (unstable_size, batch, c, h, w)
                    patches = last_A.patches

                    # tmp_weight has shape (c,), it will be applied on the (c,) dimension.
                    patches = patches * tmp_weight.view(*([1] * (patches.ndim - 3)), -1, 1, 1)  # Match with sparse or non-sparse patches.
                    next_A = last_A.create_similar(patches)

                    # bias to size (c,), need expansion before unfold.
                    bias = tmp_bias.view(-1,1,1).expand(self.input_shape[1:]).unsqueeze(0)
                    # Unfolded bias has shape (1, out_h, out_w, in_c, H, W).
                    bias_unfolded = inplace_unfold(bias, kernel_size=last_A.patches.shape[-2:], padding=last_A.padding, stride=last_A.stride,
                            inserted_zeros=last_A.inserted_zeros, output_padding=last_A.output_padding)
                    if last_A.unstable_idx is not None:
                        # Sparse bias has shape (unstable_size, batch, in_c, H, W).
                        bias_unfolded = bias_unfolded[:, last_A.unstable_idx[1], last_A.unstable_idx[2]]
                        sum_bias = torch.einsum('bschw,sbchw->sb', bias_unfolded, last_A.patches)
                        # Output sum_bias has shape (unstable_size, batch).
                    else:
                        # Patch has dimension (out_c, batch, out_h, out_w, c, h, w).
                        sum_bias = torch.einsum('bijchw,sbijchw->sbij', bias_unfolded, last_A.patches)
                        # Output sum_bias has shape (out_c, batch, out_h, out_w).
                else:
                    # we should create a real identity Patch
                    num_channel = tmp_weight.numel()
                    # desired Shape is (c, batch, out_w, out_h, c, 1, 1) or (unstable_size, batch, c, 1, 1).
                    patches = (torch.eye(num_channel, device=tmp_weight.device) * tmp_weight.view(-1)).view(num_channel, 1, 1, 1, num_channel, 1, 1)
                    # Expand out_h, out_w dimensions but not for batch dimension.
                    patches = patches.expand(-1, -1, last_A.output_shape[2], last_A.output_shape[3], -1, 1, 1)
                    if last_A.unstable_idx is not None:
                        # Select based on unstable indices.
                        patches = patches[last_A.unstable_idx[0], :, last_A.unstable_idx[1], last_A.unstable_idx[2]]
                    # Expand the batch dimension.
                    patches = patches.expand(-1, last_A.shape[1], *([-1] * (patches.ndim - 2)))
                    next_A = last_A.create_similar(patches, stride=1, padding=0, identity=0)
                    if last_A.unstable_idx is not None:
                        # Need to expand the bias and choose the selected ones.
                        bias = tmp_bias.view(-1,1,1,1).expand(-1, 1, last_A.output_shape[2], last_A.output_shape[3])
                        bias = bias[last_A.unstable_idx[0], :, last_A.unstable_idx[1], last_A.unstable_idx[2]]
                        # Expand the batch dimension, and final output shape is (unstable_size, batch).
                        sum_bias = bias.expand(-1, last_A.shape[1])
                    else:
                        # Output sum_bias has shape (out_c, batch, out_h, out_w).
                        sum_bias = tmp_bias.view(-1, 1, 1, 1).expand(-1, *last_A.shape[1:4])
            else:
                raise NotImplementedError()
            return next_A, sum_bias

        lA, lbias = _bound_oneside(last_lA)
        uA, ubias = _bound_oneside(last_uA)

        return [(lA, uA), (None, None), (None, None), (None, None), (None, None)], lbias, ubias

    def interval_propagate(self, *v):
        assert not self.is_input_perturbed(1) and not self.is_input_perturbed(2), \
            'Weight perturbation is not supported for BoundBatchNormalization'

        h_L, h_U = v[0]
        weight, bias = v[1][0], v[2][0]

        mid = (h_U + h_L) / 2.0
        diff = (h_U - h_L) / 2.0

        # Use `mid` in IBP to compute mean and variance for BN.
        # In this case, `forward` should not have been called.
        if self.bn_mode == 'ibp' and not hasattr(self, 'forward_value'):
            m, v, w, b = tuple(self.inputs[i].forward() for i in range(1, 5))
            self.forward(mid, m, v, w, b)

        if not self.training:
            assert not (self.is_input_perturbed(3) or self.is_input_perturbed(4))
            self.current_mean = v[3][0]
            self.current_var = v[4][0]
        self._check_unused_mean_or_var()
        if not self.use_affine:
            weight = torch.ones_like(weight)
            bias = torch.zeros_like(bias)

        tmp_weight = weight / torch.sqrt(self.current_var + self.eps)
        tmp_weight_abs = tmp_weight.abs()
        tmp_bias = bias - self.current_mean * tmp_weight
        shape = (1, -1) + (1,) * (mid.ndim - 2)

        # interval_propagate() of the Linear layer may encounter input with different norms.
        norm, eps = Interval.get_perturbation(v[0])[:2]
        if norm == torch.inf:
            center = tmp_weight.view(*shape) * mid + tmp_bias.view(*shape)
            deviation = tmp_weight_abs.view(*shape) * diff
        elif norm > 0:
            mid = v[0][0]
            center = tmp_weight.view(*shape) * mid + tmp_bias.view(*shape)
            if norm == 2:
                ptb = copy.deepcopy(v[0].ptb)
                ptb.eps = eps * tmp_weight_abs.max()
                return Interval(center, center, ptb=ptb)
            else:
                # General Lp norm.
                center = tmp_weight.view(*shape) * mid
                deviation = tmp_weight_abs.view(*shape) * eps  # use a Linf ball to replace Lp norm
        else:
            raise NotImplementedError

        lower, upper = center - deviation, center + deviation

        return lower, upper

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        # e.g., last layer input gurobi vars (3,32,32)
        gvars_array = np.array(v[0])
        # pre_layer_shape (1,3,32,32)
        pre_layer_shape = np.expand_dims(gvars_array, axis=0).shape
        # this layer shape (1,8,16,16)
        this_layer_shape = self.output_shape

        weight, bias = v[1], v[2]

        self.current_mean = v[3]
        self.current_var = v[4]
        self._check_unused_mean_or_var()
        if not self.use_affine:
            weight = torch.ones_like(weight)
            bias = torch.zeros_like(bias)

        tmp_bias = bias - self.current_mean / torch.sqrt(self.current_var + self.eps) * weight
        tmp_weight = weight / torch.sqrt(self.current_var + self.eps)

        new_layer_gurobi_vars = []
        neuron_idx = 0
        for out_chan_idx in range(this_layer_shape[1]):
            out_chan_vars = []
            for out_row_idx in range(this_layer_shape[2]):
                out_row_vars = []
                for out_col_idx in range(this_layer_shape[3]):
                    # print(this_layer_bias.shape, out_chan_idx, out_lbs.size(1))
                    lin_expr = tmp_bias[out_chan_idx].item() + tmp_weight[out_chan_idx].item() * gvars_array[out_chan_idx, out_row_idx, out_col_idx]
                    var = model.addVar(lb=-float('inf'), ub=float('inf'),
                                            obj=0, vtype=grb.GRB.CONTINUOUS,
                                            name=f'lay{self.name}_{neuron_idx}')
                    model.addConstr(lin_expr == var, name=f'lay{self.name}_{neuron_idx}_eq')
                    neuron_idx += 1

                    out_row_vars.append(var)
                out_chan_vars.append(out_row_vars)
            new_layer_gurobi_vars.append(out_chan_vars)

        self.solver_vars = new_layer_gurobi_vars
        model.update()

    def update_requires_input_bounds(self):
        self._check_weight_perturbation()


class LayerNormImpl(nn.Module):
    def __init__(self, axis, epsilon):
        super().__init__()
        self.axis = axis
        self.epsilon = epsilon

    def forward(self, x, scale, bias):
        mean = x.mean(self.axis, keepdim=True)
        d = x - mean
        dd = d**2
        var = dd.mean(self.axis, keepdim=True)
        var_eps = var + self.epsilon
        std_dev = torch.sqrt(var_eps)
        inv_std_dev = torch.reciprocal(std_dev)
        normalized = d * inv_std_dev
        normalized_scaled = normalized * scale + bias
        return normalized_scaled


class BoundLayerNormalization(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.complex = True
        self.model = LayerNormImpl(self.attr['axis'], self.attr['epsilon'])

    def forward(self, x, scale, bias):
        self.input = (x, scale, bias)
        return self.model(x, scale, bias)
