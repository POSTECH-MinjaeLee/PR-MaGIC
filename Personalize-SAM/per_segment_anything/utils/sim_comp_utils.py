from .img_utils import prepare_mask 

import torch 
from torch.nn import functional as F


def avg_max_mean_target_feat(ref_feat, ref_mask):
    """Agregates target features 
    * average of max and mean features.
    """
    target_feat = ref_feat[ref_mask > 0] # [n, c] = [sum(ref_mask > 0), 256] 
    target_feat_mean = target_feat.mean(0)
    target_feat_max = torch.max(target_feat, dim=0)[0] # [1, c] (pooling all the image features)
    target_feat = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)  # [1, 1, c]   
    return target_feat


def max_target_feat(ref_feat, ref_mask): 
    """Agregates target features 
    * max features.
    """
    target_feat = ref_feat[ref_mask > 0] # [n, c] = [sum(ref_mask > 0), 256]
    target_feat_max = torch.max(target_feat, dim=0)[0] # [1, c] (pooling all the image features)
    target_feat = target_feat_max.unsqueeze(0) # [1, c=256] 
    return target_feat
     

def mean_target_feat(ref_feat, ref_mask):
    """Agregates target features 
    * mean features.
    """
    target_feat = ref_feat[ref_mask > 0] # [n, c] = [sum(ref_mask > 0), 256]
    target_feat_mean = target_feat.mean(0) # [1, c] (pooling all the image features)
    target_feat = target_feat_mean.unsqueeze(0) # [1, c=256] 
    return target_feat



def _compute_ref_target_cosim(model, 
                             ref_image, ref_mask, 
                             processor, 
                             aggregate_func= avg_max_mean_target_feat):
    """Computes cosine similarities between target features and reference image features
    """
    
    # obtain device
    device = next(model.parameters()).device
    
    # Step 1: Image features encoding
    inputs = processor(images=ref_image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values

    vision_enc_output =  model.vision_encoder(pixel_values=pixel_values) 
    image_embeddings = vision_enc_output.last_hidden_state
    ref_feat = image_embeddings.squeeze().permute(1, 2, 0)
        
    # Step 2: interpolate reference mask
    ref_mask = prepare_mask(ref_mask) # 3, 1024, 1024
    ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear") # 1, 3, 64, 64
    ref_mask = ref_mask.squeeze()[0] # 64, 64

    # Step 3: Target feature extraction
    # Target feature in the reference image feature (F_{I})
    target_feat = aggregate_func(ref_feat, ref_mask) # [1, c] 

    # Step 4: cosine similarity
    h, w, C = ref_feat.shape # 64, 64, C=256
    target_feat = target_feat / target_feat.norm(dim=-1, keepdim=True)
    ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
    ref_feat = ref_feat.permute(2, 0, 1).reshape(C, h * w)
    sim = target_feat @ ref_feat # 1, 256 @ 256, h*w = 1, h*w

    sim = sim.reshape(1, 1, h, w) # 1, 1, h, w = 1, 1, 64, 64
    sim = F.interpolate(sim, scale_factor=4, mode="bilinear") # 1, 1, 256, 256; 
    sim = processor.post_process_masks(sim.unsqueeze(1),
                                    original_sizes=inputs["original_sizes"].tolist(),
                                    reshaped_input_sizes=inputs["reshaped_input_sizes"].tolist(),
                                    binarize=False)
    sim = sim[0].squeeze() # original shape = (384, 288) <- cvc example
    
    return {
        'image_embeddings': image_embeddings, # [batch_size, c, h, w] = [1, 256, 64, 64]
        'sim': sim, # originalimage shape 
        'ref_mask_feat': ref_mask, # [h, w] = [64, 64] 
        'target_feat': target_feat, # [1, c] = [1, 256]
        'ref_feat': ref_feat.reshape(C, h, w).permute(1, 2, 0), # (h, w, C)
        'vision_enc_output': vision_enc_output 
    }
    

def compute_ref_target_cosim(model, 
                           ref_image, ref_mask, 
                           processor, 
                           aggregate_func= avg_max_mean_target_feat,
                           is_train:bool= False):
    
    if is_train: 
        model.train()
        ret = _compute_ref_target_cosim(model, 
                                        ref_image, ref_mask, 
                                        processor, 
                                        aggregate_func= aggregate_func)
        return ret
    
    with torch.no_grad():
        model.eval()
        ret = _compute_ref_target_cosim(model, 
                                        ref_image, ref_mask, 
                                        processor, 
                                        aggregate_func= aggregate_func)
    return ret