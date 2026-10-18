from calendar import c
from math import e
import torch
import torch.nn.functional as F
import copy
from torch.cuda import amp
from tools.loss import diversity_loss
import numpy as np
from collections import Counter

from .cluster import get_cluster_results,generate_cluster_features,extract_features
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import torch.nn.functional as F
import torch.nn as nn
import random

def build_refined_memory_and_labels(i2r_clean, i2r_err,
                                    cluster_features_ir, cluster_features_rgb,
                                    features_ir, pseudo_labels_ir, dataset_train_ir,
                                    features_rgb, pseudo_labels_rgb, dataset_train_rgb,mode=True):

    refined_ir_centers = []
    refined_rgb_centers = []
    ir2rgb_map = {}
    ir_new2old = {}
    new_pseudo_labels_ir = [-1] * len(pseudo_labels_ir)
    new_pseudo_labels_rgb = [-1] * len(pseudo_labels_rgb)

    new_pseudo_type_ir = [-1] * len(pseudo_labels_ir)
    new_pseudo_type_rgb = [-1] * len(pseudo_labels_rgb)

    new_pseudo_soft_ir2rgb = torch.zeros(len(pseudo_labels_ir), len(cluster_features_rgb))
    new_pseudo_soft_ir2ir = torch.zeros(len(pseudo_labels_ir), len(cluster_features_rgb))
    new_pseudo_soft_rgb2ir = torch.zeros(len(pseudo_labels_rgb), len(cluster_features_rgb))
    new_pseudo_soft_rgb2rgb = torch.zeros(len(pseudo_labels_rgb), len(cluster_features_rgb))

    for ir, rgb in i2r_clean.items():
        refined_ir_centers.append(cluster_features_ir[ir])
        refined_rgb_centers.append(cluster_features_rgb[rgb])
        idx_new = len(refined_ir_centers) - 1
        ir2rgb_map[idx_new] = rgb
        ir_new2old[idx_new] = ir

        for i, lab in enumerate(pseudo_labels_ir):
            if lab == ir:
                new_pseudo_labels_ir[i] = idx_new
                new_pseudo_type_ir[i] = 0

        for i, lab in enumerate(pseudo_labels_rgb):
            if lab == rgb:
                new_pseudo_labels_rgb[i] = idx_new
                new_pseudo_type_rgb[i] = 0

    if mode == False:
        refined_ir_centers = torch.stack(refined_ir_centers, dim=0)
        refined_rgb_centers = torch.stack(refined_rgb_centers, dim=0)
        return refined_ir_centers, refined_rgb_centers, ir2rgb_map, ir_new2old, new_pseudo_labels_ir, new_pseudo_labels_rgb,new_pseudo_type_ir, new_pseudo_type_rgb

    rgb2new = {}

    for ir_class, rgb_classes in i2r_err.items():
        ir_sample_idx = [idx for idx, label in enumerate(pseudo_labels_ir) if label == ir_class]
        if not ir_sample_idx:
            continue
        ir_features_subset = features_ir[ir_sample_idx]
        rgb_centers = [cluster_features_rgb[rgb_class] for rgb_class in rgb_classes]

        refined = refine_ir_class(ir_features_subset, rgb_centers, rgb_classes)

        if len(refined) == 1:

            ir_center, rgb_id = refined[0]
            refined_ir_centers.append(ir_center)
            refined_rgb_centers.append(cluster_features_rgb[rgb_id])
            idx_new = len(refined_ir_centers) - 1
            ir2rgb_map[idx_new] = rgb_id
            ir_new2old[idx_new] = ir_class
            for j in ir_sample_idx:
                new_pseudo_labels_ir[j] = idx_new
                new_pseudo_type_ir[j] = 1
            for i, lab in enumerate(pseudo_labels_rgb):
                if lab == rgb_id:
                    new_pseudo_labels_rgb[i] = idx_new
                    new_pseudo_type_rgb[i] = 1
        else:

            ir_features_subset = F.normalize(ir_features_subset, dim=1)
            rgb_centers = F.normalize(torch.stack(rgb_centers, dim=0), dim=1)
            similarity = torch.mm(ir_features_subset, rgb_centers.T)
            assignments = similarity.argmax(dim=1)

            soft_labels = torch.softmax(similarity, dim=1)  # [N, K]

            for k, (ir_center, rgb_id) in enumerate(refined):
                refined_ir_centers.append(ir_center)
                refined_rgb_centers.append(cluster_features_rgb[rgb_id])
                idx_new = len(refined_ir_centers) - 1
                ir2rgb_map[idx_new] = rgb_id
                ir_new2old[idx_new] = ir_class
                rgb2new[rgb_id] = idx_new

                sub_idx = [ir_sample_idx[i] for i in range(len(ir_sample_idx)) if assignments[i] == k]
                for j in sub_idx:
                    new_pseudo_labels_ir[j] = idx_new
                    new_pseudo_type_ir[j] = 1

                for i, lab in enumerate(pseudo_labels_rgb):
                    if lab == rgb_id:
                        new_pseudo_labels_rgb[i] = idx_new
                        new_pseudo_type_rgb[i] = 1

            for local_i, j in enumerate(ir_sample_idx):

                probs = soft_labels[local_i]  # [K]
                for m, rgb_id_m in enumerate(rgb_classes):
                    new_idx_m = rgb2new[rgb_id_m]  # 映射到新类空间
                    new_pseudo_soft_ir2rgb[j, new_idx_m] = probs[m].item()

            new_ir_centers = [refined_ir_centers[rgb2new[rgb_id]] for rgb_id in rgb_classes if rgb_id in rgb2new]
            new_ir_centers = F.normalize(torch.stack(new_ir_centers, dim=0), dim=1)  # [K, D]

            sim_ir2ir = torch.mm(F.normalize(ir_features_subset, dim=1), new_ir_centers.T)  # [N, K]
            soft_ir2ir = torch.softmax(sim_ir2ir, dim=1)

            for local_i, j in enumerate(ir_sample_idx):
                for k in range(len(new_ir_centers)):
                    new_pseudo_soft_ir2ir[j, list(rgb2new.values())[k]] = soft_ir2ir[local_i, k].item()

            rgb_sample_idx = [i for i, lab in enumerate(pseudo_labels_rgb) if lab in rgb_classes]
            if rgb_sample_idx:
                rgb_features_subset = features_rgb[rgb_sample_idx]
                sim_rgb2ir = torch.mm(F.normalize(rgb_features_subset, dim=1), new_ir_centers.T)  # [M, K]
                soft_rgb2ir = torch.softmax(sim_rgb2ir, dim=1)

                for local_i, i in enumerate(rgb_sample_idx):
                    for k in range(len(new_ir_centers)):
                        new_pseudo_soft_rgb2ir[i, list(rgb2new.values())[k]] = soft_rgb2ir[local_i, k].item()

            new_rgb_centers = [refined_rgb_centers[rgb2new[rgb_id]] for rgb_id in rgb_classes if rgb_id in rgb2new]
            new_rgb_centers = F.normalize(torch.stack(new_rgb_centers, dim=0), dim=1)  # [K, D]

            if rgb_sample_idx:  # 已经在前面算过 rgb_sample_idx
                rgb_features_subset = features_rgb[rgb_sample_idx]
                sim_rgb2rgb = torch.mm(F.normalize(rgb_features_subset, dim=1), new_rgb_centers.T)  # [M, K]
                soft_rgb2rgb = torch.softmax(sim_rgb2rgb, dim=1)

                for local_i, i in enumerate(rgb_sample_idx):
                    for k in range(len(new_rgb_centers)):
                        new_pseudo_soft_rgb2rgb[i, list(rgb2new.values())[k]] = soft_rgb2rgb[local_i, k].item()

    refined_ir_centers = torch.stack(refined_ir_centers, dim=0)
    refined_rgb_centers = torch.stack(refined_rgb_centers, dim=0)

    num_refined_ir = refined_ir_centers.shape[0]
    num_refined_rgb = refined_rgb_centers.shape[0]

    new_pseudo_soft_ir2rgb = new_pseudo_soft_ir2rgb[:, :num_refined_rgb]
    new_pseudo_soft_ir2ir = new_pseudo_soft_ir2ir[:, :num_refined_ir]
    new_pseudo_soft_rgb2ir = new_pseudo_soft_rgb2ir[:, :num_refined_ir]
    new_pseudo_soft_rgb2rgb = new_pseudo_soft_rgb2rgb[:, :num_refined_rgb]

    return refined_ir_centers, refined_rgb_centers, ir2rgb_map, ir_new2old, new_pseudo_labels_ir, new_pseudo_labels_rgb,new_pseudo_type_ir, new_pseudo_type_rgb,new_pseudo_soft_ir2rgb,new_pseudo_soft_ir2ir,new_pseudo_soft_rgb2ir,new_pseudo_soft_rgb2rgb

def refine_ir_class(ir_features, rgb_centers, rgb_classes):
    """
    根据IR样本与RGB类中心的分配情况，决定是合并还是拆分
    ir_features: Tensor [N, D] 当前IR类的样本特征
    rgb_centers: List[Tensor] 对应的RGB类中心
    rgb_classes: List[int] 对应的RGB类id
    return: List of (new_ir_center, rgb_id)
    """

    ir_features = F.normalize(ir_features, dim=1)
    rgb_centers = F.normalize(torch.stack(rgb_centers, dim=0), dim=1)

    similarity = torch.mm(ir_features, rgb_centers.T)
    assignments = similarity.argmax(dim=1)  # [N]

    results = []
    unique_assignments = assignments.unique().tolist()

    if len(unique_assignments) == 1:

        k = unique_assignments[0]
        ir_center = ir_features.mean(0)  # 整体IR类的中心
        results.append((ir_center, rgb_classes[k]))
    elif len(unique_assignments) == 2:

        for k in unique_assignments:
            idx = (assignments == k).nonzero(as_tuple=True)[0]
            ir_center = ir_features[idx].mean(0)
            results.append((ir_center, rgb_classes[k]))
    else:

        ir_center = ir_features.mean(0)
        results.append((ir_center, None))
    return results

def build_clean_memory_and_labels(i2r_clean, i2r_err,
                                    cluster_features_ir, cluster_features_rgb,
                                    features_ir, pseudo_labels_ir, dataset_train_ir,
                                    features_rgb, pseudo_labels_rgb, dataset_train_rgb):

    refined_ir_centers = []
    refined_rgb_centers = []
    ir2rgb_map = {}
    ir_new2old = {}
    new_pseudo_labels_ir = [-1] * len(pseudo_labels_ir)
    new_pseudo_labels_rgb = [-1] * len(pseudo_labels_rgb)

    new_pseudo_type_ir = [-1] * len(pseudo_labels_ir)
    new_pseudo_type_rgb = [-1] * len(pseudo_labels_rgb)

    for ir, rgb in i2r_clean.items():
        refined_ir_centers.append(cluster_features_ir[ir])
        refined_rgb_centers.append(cluster_features_rgb[rgb])
        idx_new = len(refined_ir_centers) - 1
        ir2rgb_map[idx_new] = rgb
        ir_new2old[idx_new] = ir

        for i, lab in enumerate(pseudo_labels_ir):
            if lab == ir:
                new_pseudo_labels_ir[i] = idx_new
                new_pseudo_type_ir[i] = 0

        for i, lab in enumerate(pseudo_labels_rgb):
            if lab == rgb:
                new_pseudo_labels_rgb[i] = idx_new
                new_pseudo_type_rgb[i] = 0

    refined_ir_centers = torch.stack(refined_ir_centers, dim=0)
    refined_rgb_centers = torch.stack(refined_rgb_centers, dim=0)

    return refined_ir_centers, refined_rgb_centers, ir2rgb_map, ir_new2old, new_pseudo_labels_ir, new_pseudo_labels_rgb,new_pseudo_type_ir, new_pseudo_type_rgb

def split_ir_class(ir_features, rgb_centers):
    """
    ir_features: Tensor [N, D] 当前模糊IR类的样本特征
    rgb_centers: List[Tensor] 对应的多个RGB类中心 [K, D]
    return: 新的IR子类中心列表
    """

    ir_features = F.normalize(ir_features, dim=1)
    rgb_centers = F.normalize(torch.stack(rgb_centers, dim=0), dim=1)

    similarity = torch.mm(ir_features, rgb_centers.T)

    assignments = similarity.argmax(dim=1)  # [N]

    new_ir_centers = []
    for k in range(len(rgb_centers)):
        idx = (assignments == k).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            new_ir_centers.append(ir_features[idx].mean(0))
    return new_ir_centers

def split_from_r2i(r2i):

    ir2rgbs = defaultdict(list)
    for rgb, ir in r2i.items():
        ir2rgbs[ir].append(rgb)

    i2r_clean, r2i_clean = {}, {}
    i2r_err, r2i_err = {}, {}

    for ir, rgbs in ir2rgbs.items():
        if len(rgbs) == 1:

            rgb = rgbs[0]
            i2r_clean[ir] = rgb
            r2i_clean[rgb] = ir
        else:

            i2r_err[ir] = rgbs
            for rgb in rgbs:
                r2i_err[rgb] = ir

    return i2r_clean, r2i_clean, i2r_err, r2i_err

def match_prototypes(f_seq_rgb, f_seq_ir):
    """Match visible and infrared prototypes for PGUR refinement."""
    i2r = {}
    r2i = {}
    r2i_p = {}
    R = []
    bgm = False
    num_cluster_ir = len(f_seq_ir)
    num_cluster_rgb = len(f_seq_rgb)

    switch = False
    if num_cluster_rgb < num_cluster_ir:
        f_seq_rgb, f_seq_ir = f_seq_ir, f_seq_rgb
        switch = True
    print("Progressive Graph Matching...")

    if num_cluster_rgb >= num_cluster_ir or switch:

        f_seq_rgb = F.normalize(f_seq_rgb, dim=1)
        f_seq_ir = F.normalize(f_seq_ir, dim=1)
        similarity = ((torch.mm(f_seq_rgb, f_seq_ir.T)) / 1).exp().cpu()
        dis_similarity = (1 / (similarity))
        cost = dis_similarity / 1
        tmp = torch.zeros(dis_similarity.shape[0], dis_similarity.shape[0] - dis_similarity.shape[1])
        cost = (torch.cat((cost, tmp), 1))
        unmatched_row = []
        row_ind, col_ind = linear_sum_assignment(cost)
        for idx, item in enumerate(row_ind):
            if col_ind[idx] < similarity.shape[1]:
                R.append((row_ind[idx], col_ind[idx]))
                r2i[row_ind[idx]] = col_ind[idx]
                i2r[col_ind[idx]] = row_ind[idx]
            else:
                unmatched_row.append(row_ind[idx])

        unmatched_rows = []
        for i in range(0, len(unmatched_row), num_cluster_ir):
            unmatched_rows.append(unmatched_row[i:i + num_cluster_ir])

        if bgm is False:
            for unmatched_row in unmatched_rows:
                unmatched_cost = cost[unmatched_row][:, :dis_similarity.shape[1]]
                unmatched_row_ind, unmatched_col_ind = linear_sum_assignment(unmatched_cost)
                for idx, item in enumerate(unmatched_row_ind):
                    R.append((unmatched_row[idx], unmatched_col_ind[idx]))
                    r2i[unmatched_row[idx]] = unmatched_col_ind[idx]
                    r2i_p[unmatched_row[idx]] = unmatched_col_ind[idx]
    else:
        raise ValueError(
            f'num_cluster_rgb({num_cluster_rgb}) must be greater than or equal to num_cluster_ir({num_cluster_ir})')
    print("Progressive Graph Matching Done")

    if switch:
        i2r, r2i = r2i, i2r
    return i2r, r2i, r2i_p

def check_knn_consistency(ir_features, rgb_features, seq_ir_features, seq_rgb_features,
                          k_frame=5, k_seq=3, threshold=0.5):
    """
    ir_features: IR 类的所有帧特征 (N_ir, d)
    rgb_features: RGB 类的所有帧特征 (N_rgb, d)
    seq_ir_features: IR 类的序列特征 (1, d)
    seq_rgb_features: RGB 类的序列特征 (1, d)
    k_frame: 帧级 KNN 的 k
    k_seq: 序列级 KNN 的 k
    threshold: 帧/序列一致性阈值（如 0.5 表示 Top-k 中有一半重叠）
    """
    import torch.nn.functional as F
    ir_features = F.normalize(ir_features, dim=1)
    rgb_features = F.normalize(rgb_features, dim=1)
    seq_ir_features = F.normalize(seq_ir_features, dim=1)
    seq_rgb_features = F.normalize(seq_rgb_features, dim=1)

    sim_frame = torch.mm(ir_features, rgb_features.T)
    topk_frame = sim_frame.topk(k_frame, dim=1).indices.flatten().tolist()

    sim_seq = torch.mm(seq_ir_features, seq_rgb_features.T)
    topk_seq = sim_seq.topk(k_seq, dim=1).indices.flatten().tolist()

    inter = len(set(topk_frame) & set(topk_seq))
    ratio = inter / min(len(topk_frame), len(topk_seq))

    return ratio >= threshold

def select_merge_data(dists):
    dists = torch.from_numpy(dists)
    print(dists.size())
    dists = dists.numpy()
    ind = np.unravel_index(np.argsort(dists, axis=None)[::-1], dists.shape) #np.argsort(dists, axis=1)#
    idx1 = ind[0]
    idx2 = ind[1]
    dist_list = dists[idx1,idx2]
    return idx1, idx2, dist_list

def reshape_video_frames(x1,seq_len=6, shuffle_mode=None):
    b, c, h, w = x1.size()
    x1 = x1.view(int(b * seq_len), int(c / seq_len), h, w)
    if shuffle_mode is not None:
        x1 = x1.view(b, seq_len, c // seq_len, h, w)
        x1 = shuffle_seq_all(x1)  # 每个样本都做
        x1 = x1.view(b * seq_len, int(c / seq_len), h, w)
    return x1

def shuffle_seq_all_batchmode(x, probs=(0.5, 0.3, 0.2)):
    modes = ["swap2", "reverse", "permute"]
    mode = random.choices(modes, weights=probs, k=1)[0]
    return shuffle_seq_all(x, mode=mode)  # 调用你原来的实现

def shuffle_seq_all(x, mode="swap2"):
    """
    x: [B, T, C, H, W]
    mode: "swap2" | "permute" | "reverse"
    return: same shape, every sample is shuffled
    """
    B, T, C, H, W = x.shape
    device = x.device

    if mode == "reverse":
        return x.flip(dims=[1])  # 倒序

    idx = torch.arange(T, device=device).unsqueeze(0).repeat(B, 1)

    if mode == "permute":

        perm = torch.rand(B, T, device=device).argsort(dim=1)
        idx = perm

    elif mode == "swap2":

        batch = torch.arange(B, device=device)
        i = torch.randint(0, T, (B,), device=device)
        j = torch.randint(0, T - 1, (B,), device=device)
        j = j + (j >= i).long()  # 保证 j != i

        tmp = idx[batch, i].clone()
        idx[batch, i] = idx[batch, j]
        idx[batch, j] = tmp

    else:
        raise ValueError(f"Unknown mode: {mode}")

    idx_expand = idx.view(B, T, 1, 1, 1).expand(-1, -1, C, H, W)
    return x.gather(dim=1, index=idx_expand)

def train_ciw(iter_rgb,iter_ir,base,meter,scaler,current_epoch):
    """Causal Intervention Warm-up: MPB, TTB and ICS with baseline loss ordering."""
    for _ in range(base.steps_per_epoch):
        input1,input1_do, label1, = iter_rgb.next_one()
        input2,input2_do, label2 = iter_ir.next_one()

        base.model_optimizer.zero_grad()
        rgb_imgs, rgb_imgs_do, rgb_pids = input1,input1_do, label1
        ir_imgs,ir_imgs_do, ir_pids = input2,input2_do, label2
        rgb_imgs, rgb_imgs_do, rgb_pids = rgb_imgs.to(base.device),rgb_imgs_do.to(base.device),rgb_pids.to(base.device).long()
        ir_imgs, ir_imgs_do, ir_pids = ir_imgs.to(base.device),ir_imgs_do.to(base.device), ir_pids.to(base.device).long()
        if current_epoch <0:
            rgb_imgs_do = rgb_imgs
            ir_imgs_do = ir_imgs
            rgb_imgs_do = reshape_video_frames(rgb_imgs_do,base.sequence_length,shuffle_mode="swap2")
            ir_imgs_do = reshape_video_frames(ir_imgs_do,base.sequence_length,shuffle_mode="swap2")
        else:
            rgb_imgs_do = reshape_video_frames(rgb_imgs_do,base.sequence_length)
            ir_imgs_do = reshape_video_frames(ir_imgs_do,base.sequence_length)

            rgb_imgs_do_T = copy.deepcopy(rgb_imgs)
            ir_imgs_do_T = copy.deepcopy(ir_imgs)

            rgb_imgs_do_T = reshape_video_frames(rgb_imgs_do_T,base.sequence_length,shuffle_mode="swap2")
            ir_imgs_do_T  = reshape_video_frames(ir_imgs_do_T,base.sequence_length,shuffle_mode="swap2")

        rgb_imgs = reshape_video_frames(rgb_imgs,base.sequence_length)
        ir_imgs = reshape_video_frames(ir_imgs,base.sequence_length)

        squence_pids_set = torch.arange(1, 17)
        squence_pids = squence_pids_set.repeat_interleave(6)
        squence_pids = squence_pids.to(base.device)

        with amp.autocast(enabled=True):
            features, features_frame = base.model(x1=rgb_imgs, x2=ir_imgs)
            features_do, features_frame_do = base.model(x1=rgb_imgs_do, x2=ir_imgs_do)
            features_T, features_frame_T = base.model(x1=rgb_imgs_do_T, x2=ir_imgs_do_T)

            pids = torch.cat([rgb_pids, ir_pids], dim=0)

            triplet_loss_last = base.tri_creiteron(features_frame[0].squeeze(), squence_pids)
            triplet_loss = base.tri_creiteron(features_frame[1].squeeze(), squence_pids)
            triplet_loss_proj = base.tri_creiteron(features_frame[2].squeeze(), squence_pids)
            triplet_loss_stp = base.tri_creiteron(features_frame[3].squeeze(), squence_pids)

            intervention_loss = base.intervention_creiteron(features[0].squeeze(), features_do[0].squeeze())
            + base.intervention_creiteron(features[1].squeeze(), features_do[1].squeeze())

            intervention_loss_T = base.intervention_creiteron(features[0].squeeze(), features_T[0].squeeze())
            + base.intervention_creiteron(features[1].squeeze(), features_T[1].squeeze())

            all_triplet_loss = triplet_loss

            total_loss =  base.mpb_weight*intervention_loss + base.ttb_weight*intervention_loss_T + base.ics_weight*all_triplet_loss
        scaler.scale(total_loss).backward()
        scaler.step(base.model_optimizer)
        scaler.update()

        meter.update({

                    'all_triplet_loss': all_triplet_loss.data,
                      'intervention_loss': intervention_loss.data,
                      'intervention_loss_T': intervention_loss_T.data,
                      })
    return meter

def train_video(iter,base,meter,scaler):

    for _ in range(base.steps_per_epoch):
        input1, input2, label1, label2 = iter.next_one()
        base.model_optimizer.zero_grad()
        rgb_imgs, rgb_pids = input1, label1
        ir_imgs, ir_pids = input2, label2
        rgb_imgs,  rgb_pids = rgb_imgs.to(base.device),rgb_pids.to(base.device).long()
        ir_imgs, ir_pids = ir_imgs.to(base.device), ir_pids.to(base.device).long()
        rgb_imgs = reshape_video_frames(rgb_imgs,base.sequence_length)
        ir_imgs = reshape_video_frames(ir_imgs,base.sequence_length)

        with amp.autocast(enabled=True):
            features, cls_score = base.model(x1=rgb_imgs, x2=ir_imgs)

            pids = torch.cat([rgb_pids, ir_pids], dim=0)

            ide_loss = base.pid_creiteron(cls_score[0], pids)
            ide_loss_proj = base.pid_creiteron(cls_score[1], pids)
            ide_loss_text = base.pid_creiteron(cls_score[2], pids)
            ide_loss_cue = base.pid_creiteron(cls_score[3], pids)

            triplet_loss_last = base.tri_creiteron(features[0].squeeze(), pids)
            triplet_loss = base.tri_creiteron(features[1].squeeze(), pids)
            triplet_loss_proj = base.tri_creiteron(features[2].squeeze(), pids)
            triplet_loss_cue = base.tri_creiteron(features[3].squeeze(), pids)
            ide_loss_text = base.pgur_ir_weight*ide_loss_text
            ide_loss_cue = base.pgur_rgb_weight*ide_loss_cue
            triplet_loss_cue = base.ttb_weight*triplet_loss_cue

            total_loss = ide_loss + ide_loss_proj + triplet_loss_last + triplet_loss + triplet_loss_proj + ide_loss_text + ide_loss_cue + triplet_loss_cue
        scaler.scale(total_loss).backward()
        scaler.step(base.model_optimizer)
        scaler.update()

        meter.update({'pid_loss': ide_loss.data,
                      'pid_loss_proj': ide_loss_proj.data,
                      'tri_loss': triplet_loss.data,
                      'tri_loss_proj': triplet_loss_proj.data,
                      'tri_loss_last': triplet_loss_last.data,
                      'ide_loss_text': ide_loss_text.data,
                      'pid_loss_STP': ide_loss_cue.data,
                      'tri_loss_STP': triplet_loss_cue.data,
                      })
    return meter

def train_intra_modality(loaders, base, meter, scaler,current_epoch,logger):
    """Optimize independent visible and infrared cluster memories."""

    num_cluster_rgb, pseudo_labels_rgb, num_cluster_ir, pseudo_labels_ir, features_rgb, features_ir = get_cluster_results(
        base, loaders,
        epoch=current_epoch,
        logger=logger)

    dataset_train_ir = loaders.samples.train_ir
    dataset_train_rgb = loaders.samples.train_rgb
    features_ir = torch.cat([features_ir[f[0]].unsqueeze(0) for f, _, _ in sorted(dataset_train_ir)], 0)
    features_rgb = torch.cat([features_rgb[f[0]].unsqueeze(0) for f, _, _ in sorted(dataset_train_rgb)], 0)

    cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir)
    cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)
    base._get_ClusterMemoery(cluster_features_ir, cluster_features_rgb, num_cluster_ir, num_cluster_rgb)

    pseudo_labeled_dataset_ir = []
    ir_label = []
    pseudo_real_ir = {}
    for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_train_ir), pseudo_labels_ir)):
        if label != -1:
            pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
            ir_label.append(label.item())
    logger('==> Statistics for IR epoch {}: {} clusters'.format(current_epoch, num_cluster_ir))
    pseudo_labeled_dataset_rgb = []
    rgb_label = []
    pseudo_real_rgb = {}
    for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_train_rgb), pseudo_labels_rgb)):
        if label != -1:
            pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
            rgb_label.append(label.item())
    logger('==> Statistics for RGB epoch {}: {} clusters'.format(current_epoch, num_cluster_rgb))

    _, _ = loaders.get_cluster_loader(pseudo_labeled_dataset_rgb,pseudo_labeled_dataset_ir,rgb_label,ir_label)

    for _ in range(base.steps_per_epoch):
        input1, label1 = loaders.cluster_iter_rgb.next_one()
        input2, label2 = loaders.cluster_iter_ir.next_one()

        base.model_optimizer.zero_grad()
        rgb_imgs, rgb_pids = input1, label1
        ir_imgs, ir_pids = input2, label2
        rgb_imgs,  rgb_pids = rgb_imgs.to(base.device),rgb_pids.to(base.device).long()
        ir_imgs, ir_pids = ir_imgs.to(base.device), ir_pids.to(base.device).long()
        rgb_imgs = reshape_video_frames(rgb_imgs,base.sequence_length)
        ir_imgs = reshape_video_frames(ir_imgs,base.sequence_length)

        with amp.autocast(enabled=True):
            features, features_frame = base.model(x1=rgb_imgs, x2=ir_imgs)

            feats = torch.cat(features, dim=-1)

            B = feats.size(0)
            half = B // 2
            rgb_feat = feats[:half]  # 前一半
            ir_feat = feats[half:]

            loss_ir = base.memory_ir(ir_feat, ir_pids)
            loss_rgb = base.memory_rgb(rgb_feat, rgb_pids)

            total_loss =  loss_ir + loss_rgb

        scaler.scale(total_loss).backward()
        scaler.step(base.model_optimizer)
        scaler.update()

        meter.update({
                      'loss_ir': loss_ir.data,
                      'loss_rgb': loss_rgb.data,
                      })
    return meter

def train_pgur(loaders, base, meter, scaler,current_epoch,logger):
    """Prototype-Guided Uncertainty Refinement with hard and soft supervision."""

    num_cluster_rgb, pseudo_labels_rgb, num_cluster_ir, pseudo_labels_ir, features_rgb, features_ir = get_cluster_results(
        base, loaders,
        epoch=current_epoch,
        logger=logger)

    dataset_train_ir = loaders.samples.train_ir
    dataset_train_rgb = loaders.samples.train_rgb

    dataset_rgb_sorted = sorted(zip(dataset_train_rgb, pseudo_labels_rgb), key=lambda x: x[0][0])
    features_rgb = torch.cat([features_rgb[f[0]].unsqueeze(0) for (f, _, _), _ in dataset_rgb_sorted], 0)
    pseudo_labels_rgb = [label for (_, label) in dataset_rgb_sorted]

    dataset_ir_sorted = sorted(zip(dataset_train_ir, pseudo_labels_ir), key=lambda x: x[0][0])
    features_ir = torch.cat([features_ir[f[0]].unsqueeze(0) for (f, _, _), _ in dataset_ir_sorted], 0)
    pseudo_labels_ir = [label for (_, label) in dataset_ir_sorted]

    cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir)
    cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)

    base._get_ClusterMemoery(cluster_features_ir, cluster_features_rgb, num_cluster_ir, num_cluster_rgb)

    pseudo_labeled_dataset_ir = []
    ir_label = []

    for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_train_ir), pseudo_labels_ir)):
        if label != -1:
            pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
            ir_label.append(label.item())
    logger('==> Statistics for IR epoch {}: {} clusters'.format(current_epoch, num_cluster_ir))

    pseudo_labeled_dataset_rgb = []
    rgb_label = []

    for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_train_rgb), pseudo_labels_rgb)):
        if label != -1:
            pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
            rgb_label.append(label.item())
    logger('==> Statistics for RGB epoch {}: {} clusters'.format(current_epoch, num_cluster_rgb))

    i2r, r2i,r2i_p = match_prototypes(cluster_features_rgb, cluster_features_ir)

    if len(cluster_features_rgb) >= len(cluster_features_ir):
        i2r_clean, r2i_clean, i2r_err, r2i_err = split_from_r2i(r2i)
        logger("干净匹配 IR 数: {}".format(len(i2r_clean)))
        logger("模糊匹配 IR 数: {}".format(len(i2r_err)))
        if len(i2r_err) > 10:
            hard_data = True
        else:
            hard_data = False
    elif len(cluster_features_rgb) < len(cluster_features_ir):
        r2i_clean,i2r_clean, r2i_err,i2r_err, = split_from_r2i(i2r)
        logger("干净匹配 IR 数: {}".format(len(r2i_clean)))
        logger("模糊匹配 IR 数: {}".format(len(r2i_err)))
        if len(r2i_err)>10:
            hard_data = True
        else:
            hard_data = False

    if len(cluster_features_rgb)>len(cluster_features_ir):
        (refined_ir_centers, refined_rgb_centers, _, _,\
            new_pseudo_labels_ir, new_pseudo_labels_rgb,
         new_pseudo_type_ir, new_pseudo_type_rgb,\
         new_pseudo_soft_ir2rgb,new_pseudo_soft_ir2ir,
         new_pseudo_soft_rgb2ir,new_pseudo_soft_rgb2rgb) = build_refined_memory_and_labels(
            i2r_clean, i2r_err,
            cluster_features_ir, cluster_features_rgb,
            features_ir, pseudo_labels_ir, dataset_train_ir,
            features_rgb, pseudo_labels_rgb, dataset_train_rgb
        )
    elif len(cluster_features_rgb) < len(cluster_features_ir):

        (refined_rgb_centers, refined_ir_centers, _, _,
         new_pseudo_labels_rgb, new_pseudo_labels_ir,
         new_pseudo_type_rgb, new_pseudo_type_ir,
         new_pseudo_soft_rgb2ir, new_pseudo_soft_rgb2rgb,
         new_pseudo_soft_ir2rgb, new_pseudo_soft_ir2ir) = build_refined_memory_and_labels(
            r2i_clean, r2i_err,
            cluster_features_rgb, cluster_features_ir,
            features_rgb, pseudo_labels_rgb, dataset_train_rgb,
            features_ir, pseudo_labels_ir, dataset_train_ir
        )
    elif len(cluster_features_rgb) == len(cluster_features_ir):
        (refined_ir_centers, refined_rgb_centers, ir2rgb_map,
         ir_new2old, new_pseudo_labels_ir, new_pseudo_labels_rgb,
         new_pseudo_type_ir, new_pseudo_type_rgb) = build_refined_memory_and_labels(
            i2r_clean, i2r_err,
            cluster_features_ir, cluster_features_rgb,
            features_ir, pseudo_labels_ir, dataset_train_ir,
            features_rgb, pseudo_labels_rgb, dataset_train_rgb,mode=False
        )

    base._get_ClusterMemoery(refined_ir_centers, refined_rgb_centers,
                             refined_ir_centers.size(0), refined_rgb_centers.size(0))

    logger(f"Refine后 IR memory 大小: {refined_ir_centers.size(0)}")
    logger(f"Refine后 RGB memory 大小: {refined_rgb_centers.size(0)}")

    pseudo_labeled_dataset_ir = []
    pseudo_labeled_hard_dataset_ir = []
    ir_label = []
    ir_hard_label = []
    noise_labeled_dataset_ir = []
    noise_ir_label = []
    for (fname, _, cid), new_label,label_type in zip(sorted(dataset_train_ir), new_pseudo_labels_ir,new_pseudo_type_ir):
        if new_label != -1 and label_type == 0:
            pseudo_labeled_dataset_ir.append((fname, new_label, cid))
            ir_label.append(new_label)
        elif new_label != -1 and label_type == 1:
            pseudo_labeled_hard_dataset_ir.append((fname, new_label, cid))
            ir_hard_label.append(new_label)
        elif new_label == -1:
            noise_labeled_dataset_ir.append((fname, -1, cid))
            noise_ir_label.append(-1)

    pseudo_labeled_dataset_rgb = []
    pseudo_labeled_hard_dataset_rgb = []
    rgb_label = []
    rgb_hard_label = []
    noise_labeled_dataset_rgb = []
    noise_rgb_label_rgb = []
    for (fname, _, cid), new_label,label_type in zip(sorted(dataset_train_rgb), new_pseudo_labels_rgb,new_pseudo_type_rgb):
        if new_label != -1 and label_type == 0:
            pseudo_labeled_dataset_rgb.append((fname, new_label, cid))
            rgb_label.append(new_label)
        elif new_label != -1 and label_type == 1:
            pseudo_labeled_hard_dataset_rgb.append((fname, new_label, cid))
            rgb_hard_label.append(new_label)
        elif new_label == -1:
            noise_labeled_dataset_rgb.append((fname, -1, cid))
            noise_rgb_label_rgb.append(-1)

    unified_modal_iter = loaders.get_unified_modal_loader(pseudo_labeled_dataset_rgb, pseudo_labeled_dataset_ir, rgb_label, ir_label)
    if hard_data:
        unified_modal_hard_iter = loaders.get_hard_modal_loader(pseudo_labeled_hard_dataset_rgb, pseudo_labeled_hard_dataset_ir, rgb_hard_label, ir_hard_label)

    for _ in range(base.steps_per_epoch):
        input1, input2, label1, label2,index1,index2 = unified_modal_iter.next_one()
        if hard_data:
            hard_input1,hard_input2,hard_label1,hard_label2,indexs_hard_1,indexs_hard_2 = unified_modal_hard_iter.next_one()

        base.model_optimizer.zero_grad()
        rgb_imgs, rgb_pids = input1, label1
        ir_imgs, ir_pids = input2, label2

        rgb_imgs,  rgb_pids = rgb_imgs.to(base.device),rgb_pids.to(base.device).long()
        ir_imgs, ir_pids = ir_imgs.to(base.device), ir_pids.to(base.device).long()

        if hard_data:
            hard_rgb_imgs, hard_rgb_pids = hard_input1, hard_label1
            hard_ir_imgs, hard_ir_pids = hard_input2, hard_label2

            hard_rgb_imgs, hard_rgb_pids = hard_rgb_imgs.to(base.device), hard_rgb_pids.to(base.device)
            hard_ir_imgs,hard_ir_pids = hard_ir_imgs.to(base.device), hard_ir_pids.to(base.device)

            ir2ir_label = new_pseudo_soft_ir2ir[indexs_hard_2]
            ir2rgb_label = new_pseudo_soft_ir2rgb[indexs_hard_2]

            rgb2rgb_label = new_pseudo_soft_rgb2rgb[indexs_hard_1]
            rgb2ir_label = new_pseudo_soft_rgb2ir[indexs_hard_1]

            ir2ir_label,rgb2rgb_label = ir2ir_label.to(base.device),rgb2rgb_label.to(base.device)
            ir2rgb_label,rgb2ir_label = ir2rgb_label.to(base.device),rgb2ir_label.to(base.device)

            hard_rgb_imgs = reshape_video_frames(hard_rgb_imgs, base.sequence_length)
            hard_ir_imgs = reshape_video_frames(hard_ir_imgs, base.sequence_length)

        rgb_imgs = reshape_video_frames(rgb_imgs,base.sequence_length)
        ir_imgs = reshape_video_frames(ir_imgs,base.sequence_length)

        with amp.autocast(enabled=True):
            features, features_frame = base.model(x1=rgb_imgs, x2=ir_imgs)

            feats = torch.cat(features, dim=-1)
            B = feats.size(0)
            half = B // 2
            rgb_feat = feats[:half]  # 前一半
            ir_feat = feats[half:]

            loss_ir = base.memory_ir(ir_feat, ir_pids)
            loss_rgb = base.memory_rgb(rgb_feat, rgb_pids)
            cross_loss = base.memory_rgb(ir_feat, ir_pids) + base.memory_ir(rgb_feat,rgb_pids)

            if hard_data:
                hard_features,_ = base.model(x1=hard_rgb_imgs, x2=hard_ir_imgs)
                hard_feats = torch.cat(hard_features, dim=-1)

                hard_rgb_feat = hard_feats[:half]
                hard_ir_feat = hard_feats[half:]

                feats_ir  = base.memory_ir(hard_ir_feat, hard_ir_pids, return_out=True)
                feats_rgb = base.memory_rgb(hard_rgb_feat, hard_rgb_pids, return_out=True)

                log_probs_ir = nn.LogSoftmax(dim=1)(feats_ir)
                log_probs_rgb = nn.LogSoftmax(dim=1)(feats_rgb)

                loss_ir_hard = (- ir2ir_label * log_probs_ir).sum(1) .mean()
                loss_rgb_hard = (- rgb2rgb_label * log_probs_rgb).sum(1).mean()

                feats_ir2rgb  = base.memory_rgb(hard_ir_feat, hard_ir_pids, return_out=True)
                feats_rgb2ir = base.memory_ir(hard_rgb_feat, hard_rgb_pids, return_out=True)

                log_probs_ir2rgb = nn.LogSoftmax(dim=1)(feats_ir2rgb)
                log_probs_rgb2ir = nn.LogSoftmax(dim=1)(feats_rgb2ir)

                ir_hard_loss =  (- ir2rgb_label * log_probs_ir2rgb).sum(1) .mean() + loss_ir_hard
                rgb_hard_loss = (- rgb2ir_label * log_probs_rgb2ir).sum(1).mean() + loss_rgb_hard

                total_loss =  loss_ir + loss_rgb + cross_loss + base.pgur_ir_weight*ir_hard_loss + base.pgur_rgb_weight*rgb_hard_loss

            else:
                total_loss = loss_ir + loss_rgb + cross_loss
        scaler.scale(total_loss).backward()
        scaler.step(base.model_optimizer)
        scaler.update()

        if hard_data:
            meter.update({
                          'loss_ir': loss_ir.data,
                          'loss_rgb': loss_rgb.data,
                          'loss_cross': cross_loss.data,
                          'loss_ir_hard': ir_hard_loss.data,
                        'loss_rgb_hard': rgb_hard_loss.data,

                          })
        else:
            meter.update({
                'loss_ir': loss_ir.data,
                'loss_rgb': loss_rgb.data,
                'loss_cross': cross_loss.data,
            })
    return meter
