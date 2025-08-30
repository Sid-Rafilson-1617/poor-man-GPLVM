'''
For Roman Huszar's T-maze dataset
'''
import numpy as np
import pandas as pd
from sklearn.cluster import dbscan
import pynapple as nap

def classify_latent(map_latent,speed_tsd,speed_thresh=5,min_time_bin=10):
    '''
    classify the latent into spatial and non-spatial
        spatial -- during run, one cluster
        non-spatial -- during run multi cluster, or just stationary
    map_latent: n_time, latent label
    speed_tsd: n_time, speed
    speed_thresh: speed threshold to define run
    min_time_bin: minimum time to be considered as spatial
    '''
    is_spatial_all_latent = {}
    cluster_label_per_time_all_latent={}
    possible_latent = np.unique(map_latent)
    for latent_i in possible_latent:
        latent_ma = map_latent==latent_i
        run_ma = speed_tsd.d > speed_thresh
        ma = np.logical_and(latent_ma,run_ma)
        if ma.sum()>min_time_bin:
            tocluster=speed_tsd[ma]['x','y'].d
            core_samples, labels=dbscan(tocluster,eps=10,metric='euclidean',)
            cluster_label_per_time_all_latent[latent_i] = labels
            if set(labels)== set([-1,0]) or set(labels)== set([0]): # spatial only if one cluster /+ noise
                is_spatial_all_latent[latent_i] = True
            else: # all noise, or multi cluster
                is_spatial_all_latent[latent_i]=False
        else:
            is_spatial_all_latent[latent_i] = False
    is_spatial_all_latent=pd.Series(is_spatial_all_latent)

    position_latent = is_spatial_all_latent.loc[is_spatial_all_latent]
    nonposition_latent=is_spatial_all_latent.loc[np.logical_not(is_spatial_all_latent)]

    latent_classify_res = {'position_latent':position_latent,'nonposition_latent':nonposition_latent,'cluster_label_per_time_all_latent':cluster_label_per_time_all_latent}
    return latent_classify_res