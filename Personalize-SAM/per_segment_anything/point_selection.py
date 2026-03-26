from torch.nn import functional as F
import torch 
import numpy as np 
import cv2 


def naive_points_selection(sim, num_samples= 100, thr_quantile= 0.9): 
    # the shape of sim (similarity matrix)
    w, h = sim.shape
    
    # empirical distribution
    sim_prob_np_flat = F.softmax(sim.flatten(), dim=0).detach().cpu().numpy() 

    # sample points from the empirical distribution
    sample_xy_flat = np.random.choice(len(sim_prob_np_flat), size= num_samples, p=sim_prob_np_flat)
    sample_x = (sample_xy_flat // h)[None, :] # heght
    sample_y = (sample_xy_flat - sample_x * h) # width
    
    # coordinates
    sample_xy = np.concatenate((sample_y, sample_x), axis=0).transpose()
    sample_sim_prob_np_flat = sim_prob_np_flat[sample_xy_flat]
    q = np.quantile(sample_sim_prob_np_flat, thr_quantile)
    
    # obtain sample labels (naive labeling)
    sample_label = (sample_sim_prob_np_flat >= q).astype(np.int64)
    
    return sample_xy, sample_label


def points_selection_lang_dynamics(sim, 
                                step_size:float= 10, 
                                num_steps:int= 100, 
                                num_samples:int = 100,
                                temperature:float= 0.01,
                                thr_quantile:float= 0.90):

    h, w = sim.shape
    sim_np = sim.detach().cpu().numpy()

    # compute grad_x, grad_y
    gradient_x = cv2.Sobel(sim_np, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(sim_np, cv2.CV_64F, 0, 1, ksize=3)

    # compute the magnitude and angle of an image (similarity)
    magnitude, angle = cv2.cartToPolar(gradient_x, gradient_y, angleInDegrees=True)

    # compute normalized gradient; direction vectors
    norm_gradient_x = gradient_x / (magnitude+1e-8) 
    norm_gradient_y = gradient_y / (magnitude+1e-8)
        
    # samples from empirical distribution 
    # "quantile selection"
    # -> sample_xy
    sample_xy, sample_label = naive_points_selection(sim, num_samples=num_samples, thr_quantile= thr_quantile)
    sample_x_pos, sample_y_pos = sample_xy[:, 0], sample_xy[:, 1]

    # pseudo Langevin 
    for i in range(num_steps):
    
        # i'th step size         
        step_size_i = np.maximum(np.minimum(step_size + temperature * i, 5), 1)

        # direction towards a update
        eps_y, eps_x = np.random.randn(num_samples), np.random.randn(num_samples)
        g_y = norm_gradient_y[sample_y_pos, sample_x_pos] * step_size_i + eps_y * np.sqrt(step_size_i)
        g_x = norm_gradient_x[sample_y_pos, sample_x_pos] * step_size_i + eps_x * np.sqrt(step_size_i)

        # updated positions of negative samples
        # eps_y, eps_x = np.random.randn(2) # stochasticity
        sample_y_pos = np.ceil(np.minimum(np.maximum(sample_y_pos + g_y, 0), h-1)).astype(np.int64)
        sample_x_pos = np.ceil(np.minimum(np.maximum(sample_x_pos + g_x, 0), w-1)).astype(np.int64)


    sample_xy = np.concatenate((sample_x_pos[None,:], sample_y_pos[None,:]), axis=0).transpose()
    
    center_xy = np.ceil(np.mean(sample_xy, axis=0, keepdims= True)).astype(np.int64)
    sample_xy = np.concatenate([center_xy, sample_xy], axis=0) 
    sample_label = np.concatenate([[1], sample_label], axis=0) 
    
    return sample_xy, sample_label


# compute difference between top1_sim and label the sample points accordingly.
def label(top1_sim, sampled_sim, thr_delta):
    delta = np.abs(top1_sim - sampled_sim)
    sample_label = (np.abs(delta) <= thr_delta).astype(np.int64)
    return sample_label


def compute_bbox(center, points, x_max, y_max): 
    # center = sample_xy[0, :]
    # points = sample_xy[1:, :]
    # distances = np.sqrt(np.sum((points - center) ** 2, axis=1))
    # radius = np.mean(distances).astype(np.int64)
    vertical_distance = np.max(np.abs(points[:, 1] - center[1]))
    horizontal_distance = np.max(np.abs(points[:, 0] - center[0]))
    
    # bounding_box_width = 2 * horizontal_distance
    # bounding_box_height = 2 * vertical_distance
    
    x_tl, y_tl = np.maximum(center[0] - horizontal_distance, 0), np.maximum(center[1] - vertical_distance, 0)
    x_br, y_br = np.minimum(center[0] + horizontal_distance, x_max-1), np.minimum(center[1] + vertical_distance, y_max-1)
    
    input_box = [[x_tl, y_tl, x_br, y_br]]

    return input_box