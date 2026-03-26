# function to apply peft to the vision_encoder (sam.image_encoder)
import torch.nn.utils.parametrize as parametrize
from torch import nn 
import torch 


class LoRAParametrization(nn.Module):
    def __init__(self, features_in, features_out, rank=1, alpha=1, device='cpu'):
        super().__init__()
        
        self.lora_A = nn.Parameter(torch.zeros((rank,features_out)).to(device))
        self.lora_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))
        nn.init.normal_(self.lora_A, mean=0, std=1)
        
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, original_weights):
        if self.enabled:
            # Return W + (B*A)*scale
            return original_weights + torch.matmul(self.lora_B, self.lora_A).view(original_weights.shape) * self.scale
        else:
            return original_weights


class ve_qkv_LoRAParametrization(nn.Module):
    def __init__(self, features_in, features_out, rank=1, alpha=1, device='cpu'):
        super().__init__()
        
        self.lora_A = nn.Parameter(torch.zeros((rank,features_out)).to(device))
        self.lora_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))
        nn.init.normal_(self.lora_A, mean=0, std=1)
        
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, original_weights):
        if self.enabled:
            # Return W + (B*A)*scale
            return original_weights + torch.matmul(self.lora_B, self.lora_A).view(original_weights.shape) * self.scale
        else:
            return original_weights

        
def ve_qkv_linear_layer_parameterization(layer, device, rank=1, lora_alpha=1):
    # Only add the parameterization to the weight matrix, ignore the Bias.    
    features_in, features_out = layer.weight.shape
    return LoRAParametrization(
        features_in, features_out, rank=rank, alpha=lora_alpha, device=device
    )
    
    
def ve_attention_layer_parameterization(layer, device, rank=1, lora_alpha=1, lora_q= True, lora_k= True, lora_v= False): 
    
    if hasattr(layer, 'qkv'):
        parametrize.register_parametrization(
            layer.qkv, "weight", ve_qkv_linear_layer_parameterization(layer.qkv, layer.qkv.weight.device, rank, lora_alpha)
        )     
    
    
def apply_lora2vision_encoder_attention(model, device, rank=10, lora_alpha=1, lora_q= True, lora_k= True, lora_v= False): 

    for i in range(len(model.image_encoder.blocks)):
        layer = model.image_encoder.blocks[i].attn
        ve_attention_layer_parameterization(layer, layer.qkv.weight.device, rank, lora_alpha=lora_alpha)


def enable_disable_lora_sam_vision_encoder(model, enabled= True): 
    
    def kqv_enable(layer, enabled): 
        if hasattr(layer, 'qkv'):
            if hasattr(layer.qkv, 'parametrizations'):
                layer.qkv.parametrizations["weight"][0].enabled = enabled            
            
    for i in range(len(model.image_encoder.blocks)):
        layer = model.image_encoder.blocks[i].attn
        kqv_enable(layer, enabled)           
    
    return 

def set_lora_train_mode(model): 
            
    for name, param in model.named_parameters(): 
        if 'lora' not in name: 
            param.requires_grad = False
        else: 
            param.requires_grad = True

    return model.image_encoder


def set_lora_eval_mode(model): 
            
    for name, param in model.named_parameters(): 
        param.requires_grad = False

    return model.image_encoder
