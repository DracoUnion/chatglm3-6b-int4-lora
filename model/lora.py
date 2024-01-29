import torch
from .quantization import *


class LoraQuantizedLinear(torch.nn.Module):

    def __init__(self, q_linear, lora_r=32, lora_alpha=32, lora_dropout_rate=0.0):
        super().__init__()

        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout_rate = lora_dropout_rate
        self.weight_bit_width = q_linear.weight_bit_width
        self.weight = q_linear.weight
        self.weight_scale = q_linear.weight_scale
        self.bias = q_linear.bias

        self.weight.requires_grad = False
        self.weight_scale.requires_grad = False
        if self.bias is not None: self.bias.requires_grad = False

        out_dim, in_dim = self.weight.shape
        if self.weight_bit_width == 4: in_dim *= 2
        self.lora_a = torch.nn.Parameter(torch.empty(
            [self.lora_r, in_dim],
            device=self.weight.device,
            dtype=torch.float16,
        ))
        torch.nn.init.kaiming_normal_(self.lora_a)
        self.lora_b = torch.nn.Parameter(torch.zeros(
            [out_dim, self.lora_r],
            device=self.weight.device,
            dtype=torch.float16,
        ))
        self.lora_dropout = torch.nn.Dropout(self.lora_dropout_rate)
        self.lora_scale = self.lora_alpha / self.lora_r

    def forward(self, input):
        ori_output = QuantizedLinear.forward(self, input)
        lora_output = (
                self.lora_dropout(input.half()) @
                self.lora_a.transpose(0, 1) @
                self.lora_b.transpose(0, 1) *
                self.lora_scale
        )
        return ori_output + lora_output.to(ori_output.dtype)

    def merge(self):
        # H = XW + b + XAB * s => H = X(W + AB * s) + b
        # 将 int 原始参数转成 fp16
        weight = extract_weight_to_half(self.weight, self.weight_scale, self.weight_bit_width)
        # 合并 lora 参数
        weight += self.lora_b @ self.lora_a * self.lora_scale
        # 再转回 int
        weight, weight_scale = half_weight_to_int(weight, self.weight_bit_width)
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        # 重新初始化 lora 两个矩阵
        torch.nn.init.kaiming_normal_(self.lora_a)
        torch.nn.init.zeros_(self.lora_b)


# def attach_lora(model, lora_r=8, lora_alpha=8, lora_dropout_rate=0.0):
#     if model.lora_attached: return model
#     lora_conf = dict(lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout_rate=lora_dropout_rate)
#     for mod in model.modules():
#         for name in dir(mod):
#             submod = getattr(mod, name, None)
#             if not isinstance(submod, QuantizedLinear):
#                 continue
#             new_submod = LoraQuantizedLinear(submod, **lora_conf)
#             setattr(mod, name, new_submod)
#
#     for name, param in model.named_parameters():
#         if 'lora_' not in name:
#             param.requires_grad = False
#     model.lora_attached = True
#     return model
def attach_lora(model, lora_r=8, lora_alpha=8, lora_dropout_rate=0.0):
    # 检查 model 是否已经有 lora_attached 属性
    if hasattr(model, 'lora_attached') and model.lora_attached:
        return model

    # 接下来的代码是将 LoRA 附加到模型上
    lora_conf = dict(lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout_rate=lora_dropout_rate)
    for mod in model.modules():
        for name in dir(mod):
            submod = getattr(mod, name, None)
            if not isinstance(submod, QuantizedLinear):
                continue
            new_submod = LoraQuantizedLinear(submod, **lora_conf)
            setattr(mod, name, new_submod)

    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False

    # 在 model 上标记 lora_attached 属性
    model.lora_attached = True

    return model


def lora_state_dict(model):
    return {
        k: v
        for k, v in model.state_dict().items()
        if 'lora_' in k
    }


def base_state_dict(model):
    return {
        k: v
        for k, v in model.state_dict().items()
        if 'lora_' not in k
    }


def merge_lora(model):
    for mod in model.modules():
        if isinstance(mod, LoraQuantizedLinear):
            mod.merge()
    return model


def detach_lora(model):
    if not model.lora_attached: return model

    for mod in model.modules():
        for name in dir(mod):
            submod = getattr(mod, name, None)
            if not isinstance(submod, LoraQuantizedLinear):
                continue
            new_submod = QuantizedLinear.from_params(
                submod.weight_bit_width,
                submod.weight,
                submod.weight_scale,
                submod.bias,
            )
            setattr(mod, name, new_submod)

    model.lora_attached = False
    return model
