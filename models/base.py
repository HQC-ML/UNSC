import torch
import torch.nn as nn
from functools import partial

class Base(torch.nn.Module):
    """
    base class for models cabable of computing the activation maps
    """
    def __init__(self):
        super().__init__()
        self.features = None
        self.fc = None
        self.in_act = {}
        self.in_map_size = {}
        self.out_map_size = {}
        self.ksize = {}
        self.stride = {}
        self.in_channel = {}
        self.padding = {}
        self.dilation = {}
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                nn.init.zeros_(m.bias.data)

    def get_activation(self, name, layer, inp, out):
        self.in_act[name] = inp[0].detach()
        self.in_map_size[name] = inp[0].shape[-1] # the map size
        self.out_map_size[name] = out[0].shape[-1]

    def register_hook(self):
        for name, layer in self.features.named_modules():
            if any([isinstance(layer, nn.Conv2d),isinstance(layer, nn.Linear)]):
                layer_name = f'{layer.__class__.__name__}-features-{name}'
                layer.register_forward_hook(partial(self.get_activation, layer_name))

                if isinstance(layer, nn.Conv2d):
                    self.ksize[layer_name] = layer.kernel_size[0]
                    self.stride[layer_name] = layer.stride[0]
                    assert layer.stride[0] == layer.stride[1], 'stride must be equal'
                    assert layer.padding[0] == layer.padding[1], 'padding must be equal'
                    self.in_channel[layer_name] = layer.in_channels
                    self.padding[layer_name] = layer.padding[0]
                    self.dilation[layer_name] = layer.dilation[0]

                elif isinstance(layer, nn.Linear):
                    self.ksize[layer_name] = 0
                    self.in_channel[layer_name] = 0
        
        for name, layer in self.fc.named_modules():
            layer_name = f'{layer.__class__.__name__}-fc-{name}'
            if any([isinstance(layer, nn.Conv2d),isinstance(layer, nn.Linear)]):
                layer.register_forward_hook(partial(self.get_activation, layer_name)) 

                if isinstance(layer, nn.Conv2d):
                    self.ksize.append(layer.kernel_size[0])
                    self.ksize[layer_name] = layer.kernel_size[0]
                    self.in_channel[layer_name] = layer.in_channels
                elif isinstance(layer, nn.Linear):
                    self.ksize[layer_name] = 0
                    self.in_channel[layer_name] = 0

    def forward(self, x):
        raise NotImplementedError