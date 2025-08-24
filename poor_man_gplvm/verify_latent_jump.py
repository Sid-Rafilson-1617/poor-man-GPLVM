import numpy as np
import pandas as pd

def get_contrast_axis_and_proj(x_sub,tuning,map_state_pre,map_state_post,map_state_win=3):
    '''
    given tuning, get the difference in the population vectors between two states and normalize, 
    project the PV on this contrastive axis

    each axis is averaged within a window (-map_state_win,+map_state_win) to account for sparse firing using the similarity of adjacent states
    '''
    state_ind_pre_range = slice(map_state_pre-map_state_win,map_state_pre+map_state_win+1)
    axis_pre=tuning[state_ind_pre_range].mean(axis=0)

    state_ind_post_range = slice(map_state_post-map_state_win,map_state_post+map_state_win+1)
    axis_post=tuning[state_ind_post_range].mean(axis=0)

    axis_pre_minus_post=axis_pre-axis_post
    axis_pre_minus_post_norm = axis_pre_minus_post/np.linalg.norm(axis_pre_minus_post)
    contrast_axis=axis_pre_minus_post_norm

    proj_on_contrast_axis=x_sub.dot(contrast_axis)
    
    return proj_on_contrast_axis,contrast_axis