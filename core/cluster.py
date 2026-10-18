from sklearn.cluster import DBSCAN
from tools.meter import AverageMeter
from collections import OrderedDict
import torch
from torch.autograd import Variable
import time
from tools.faiss_rerank import compute_jaccard_dist
import numpy as np
import collections
import hdbscan

def reshape_video_frames(x1,seq_len=6):
    b, c, h, w = x1.size()
    x1 = x1.view(int(b * seq_len), int(c / seq_len), h, w)
    return x1

def tem_pool(features,t=6):
    features = features.squeeze()
    features = features.view(features.size(0)//t, t, -1).permute(1, 0, 2)
    features = features.mean(0)
    return features

def extract_features(base,logger,loader,print_freq=10):
    base.set_eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (input,fnames,label,_) in enumerate(loader):
            data_time.update(time.time() - end)

            input = Variable(input.to(base.device))
            input = reshape_video_frames(input)
            outputs = base.model(x1=input)
            for fname, output, pid in zip(fnames, outputs, label):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (batch_idx + 1) % print_freq == 0:
                logger('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(batch_idx + 1, len(loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels

def calc_eps(rerank_dist):
    tri_mat = np.triu(rerank_dist, 1)  # tri_mat.dim=2
    tri_mat = tri_mat[np.nonzero(tri_mat)]  # tri_mat.dim=1
    tri_mat = np.sort(tri_mat,axis=None)
    rho = 1.6e-3
    top_num = np.round(rho*tri_mat.size).astype(int)
    eps = tri_mat[:top_num].mean()

    return eps

@torch.no_grad()
def generate_cluster_features(labels, features):
    centers = collections.defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])

    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]

    centers = torch.stack(centers, dim=0)
    return centers

def get_cluster_results(base,loaders,epoch=0,logger=None):
    dict_f_rgb, labels = extract_features(base,logger, loaders.rgb_cluster_loader, print_freq=50)
    dict_f_ir, _ = extract_features(base,logger, loaders.ir_cluster_loader, print_freq=50)

    cf_rgb = torch.stack(list(dict_f_rgb.values()))
    cf_ir = torch.stack(list(dict_f_ir.values()))

    rgb_rerank_dist = compute_jaccard_dist(cf_rgb,base, use_gpu=True).numpy()
    ir_rerank_dist = compute_jaccard_dist(cf_ir, base,use_gpu=True).numpy()

    if epoch == 0:

        ir_eps = 0.6
        logger('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
        base.cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
        rgb_eps = 0.6
        logger('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
        base.cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)

    logger('Clustering and labeling...')
    rgb_rerank_dist = rgb_rerank_dist.astype(np.float64)
    pseudo_labels_rgb = base.cluster_rgb.fit_predict(rgb_rerank_dist)
    num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)

    ir_rerank_dist =ir_rerank_dist.astype(np.float64)
    pseudo_labels_ir = base.cluster_ir.fit_predict(ir_rerank_dist)
    num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)

    logger('\n Clustered into {} classes on rgb \n'.format(num_cluster_rgb))
    logger('\n Clustered into {} classes on ir \n'.format(num_cluster_ir))

    return num_cluster_rgb,pseudo_labels_rgb,num_cluster_ir,pseudo_labels_ir,dict_f_rgb,dict_f_ir
