"""
@Author: Du Yunhao
@Filename: dataloader.py
@Contact: dyh_bupt@163.com
@Time: 2022/8/29 19:42
@Discription: Dataloader
"""
import json
import math
import torch
import random
import numpy as np
from PIL import Image
from os.path import join
from numpy.random import choice
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from tools.utils import *
from data_loader.sampler import RandomIdentitySampler, ConsistentModalitySampler, RandomCameraSampler
import os.path as osp
from collections import defaultdict, Counter
from collections import OrderedDict

from collections import OrderedDict
import numpy as np

from collections import defaultdict, OrderedDict
import numpy as np

def _pad_to_len(frm_paths, clip_len=6, mode="repeat_last"):
    cur = len(frm_paths)
    if cur == 0:
        return frm_paths
    if cur >= clip_len:
        return frm_paths[:clip_len]

    if mode == "repeat_last":
        return frm_paths + [frm_paths[-1]] * (clip_len - cur)
    elif mode == "cycle":
        rep = (clip_len + cur - 1) // cur
        return (frm_paths * rep)[:clip_len]
    else:
        raise ValueError(f"Unknown pad mode: {mode}")

def _shift_list(xs, k):

    if len(xs) == 0:
        return xs
    k = k % len(xs)
    return xs[k:] + xs[:k]

def _decide_n_clips(frm_num, clip_len, min_clips=1, mid_clips=3, max_clips=8, short_thr=50, mid_thr=150):
    """
    你可以按数据分布调阈值：
    - <50 帧：尽量 2-3 个 view（但别太多）
    - 50-150：3-4 个
    - >150：最多 cap 到 8
    """
    if frm_num < clip_len:
        return min_clips  # 短序列额外 view 在外层处理
    if frm_num < short_thr:
        return min(mid_clips, max_clips)  # 例如 3
    if frm_num < mid_thr:
        return min(mid_clips + 1, max_clips)  # 例如 4

    approx = int(np.ceil(frm_num / clip_len))  # 980/30≈33 -> cap 到 8
    return int(np.clip(approx, mid_clips + 1, max_clips))

def split_tracklets_balanced(
    tracklets,
    clip_len=30,

    min_clips=1,
    mid_clips=3,
    max_clips=8,
    short_thr=50,
    mid_thr=150,

    short_views=2,          # 你要“2-3 个”，这里设 2 或 3
    short_view_mode="shift_cycle",  # shift_cycle 更稳定
    pad_short=True,
    pad_mode="cycle",

    debug_short_threshold=10,
    return_debug=True,
):
    new_tracklets = []
    debug = {
        "empty_tracklets": 0,
        "short_tracklets": [],           # (pid, camid, length, first_path)
        "per_pid_clip_count": defaultdict(int),
        "per_tracklet_clip_count": [],   # 每条原tracklet切了几个clips，便于看均衡性
    }

    for idx, (frm_paths, pid, camid) in enumerate(tracklets):
        frm_num = len(frm_paths)
        if frm_num == 0:
            debug["empty_tracklets"] += 1
            continue

        if frm_num < clip_len:
            if not pad_short:
                continue

            n_views = max(1, int(short_views))

            n_views = min(n_views, frm_num) if frm_num > 0 else 1

            for v in range(n_views):
                if short_view_mode == "shift_cycle":
                    shifted = _shift_list(frm_paths, v)   # v=0,1,2... 循环位移
                    clip = _pad_to_len(shifted, clip_len=clip_len, mode=pad_mode)
                else:
                    clip = _pad_to_len(frm_paths, clip_len=clip_len, mode=pad_mode)

                new_tracklets.append((clip, pid, camid))
                debug["per_pid_clip_count"][pid] += 1

            debug["per_tracklet_clip_count"].append(n_views)
            if frm_num < debug_short_threshold:
                debug["short_tracklets"].append((pid, camid, frm_num, frm_paths[0]))
            continue

        n_clips = _decide_n_clips(
            frm_num, clip_len,
            min_clips=min_clips, mid_clips=mid_clips, max_clips=max_clips,
            short_thr=short_thr, mid_thr=mid_thr
        )

        max_start = frm_num - clip_len
        if max_start < 0:

            clip = _pad_to_len(frm_paths, clip_len=clip_len, mode=pad_mode)
            new_tracklets.append((clip, pid, camid))
            debug["per_pid_clip_count"][pid] += 1
            debug["per_tracklet_clip_count"].append(1)
            continue

        if n_clips == 1:
            starts = [0]
        else:

            starts = np.linspace(0, max_start, num=n_clips, dtype=int).tolist()
            starts = sorted(set(starts))
            if starts[-1] != max_start:
                starts[-1] = max_start  # 保证最后一个 clip 对齐尾部

        for s in starts:
            clip = frm_paths[s:s + clip_len]  # 这里不会越界，因为 s<=max_start
            new_tracklets.append((clip, pid, camid))
            debug["per_pid_clip_count"][pid] += 1

        debug["per_tracklet_clip_count"].append(len(starts))

    if return_debug:
        return new_tracklets, debug
    return new_tracklets

def relabel(meta_data):
    pid_set = set()
    for _, pid, _ in meta_data:
        pid_set.add(pid)
    pid2label = {pid: label for label, pid in enumerate(list(pid_set))}

    relabel_data = []
    for i, data in enumerate(meta_data):
        label = pid2label[data[1]]
        relabel_data.append((data[0], label, data[2]))

    return relabel_data

def spilt_dense_data(tracklets, chunk_size=10):
    new_tracklets = []

    for frm_paths, pid, camid in tracklets:
        frm_num = len(frm_paths)

        for i in range(0, frm_num, chunk_size):
            if i + chunk_size > frm_num: break
            new_frm_paths = frm_paths[i:i + chunk_size]
            new_tracklets.append((new_frm_paths, pid, camid))
    return new_tracklets

def spilt_dense_data_interval(tracklets, chunk_size=10, interval=2, num_starts=None):
    """
    在全局范围内进行间隔采样，支持多起点，起点尽量分散以保证跨度。

    Args:
        tracklets: [(frm_paths, pid, camid), ...]
        chunk_size: 每个子序列的帧数
        interval: 全局采样间隔 (stride)，例如 2 表示每隔 2 帧采一个
        num_starts: 起点数量（默认等于 interval，如果帧数较多，可以手动调大）

    Returns:
        new_tracklets: [(sampled_paths, pid, camid), ...]
    """
    new_tracklets = []

    for frm_paths, pid, camid in tracklets:
        n = len(frm_paths)
        if n < chunk_size * interval:

            continue

        if num_starts is None:
            num_starts = interval

        max_start = min(n - chunk_size * interval, interval * num_starts)
        starts = np.linspace(0, max_start, num_starts, dtype=int)

        for start in starts:
            idxs = list(range(start, n, interval))  # 全局间隔采样

            for i in range(0, len(idxs) - chunk_size + 1, chunk_size):
                seg_ids = idxs[i:i + chunk_size]
                seg_paths = [frm_paths[j] for j in seg_ids]
                new_tracklets.append((seg_paths, pid, camid))

    return new_tracklets

def get_dataloader(opt, mode,transform=None,transform2=None,transform3=None, show=False):
    if mode in ('train', 'auxiliary'):
        sample_mode = opt.train_frame_sampling

    elif mode in ('query', 'gallery'):
        sample_mode = opt.test_frame_sampling

    else:
        raise RuntimeError(f'Wrong dataloader mode {mode}.')

    if opt.dataset == 'bupt':
        dataset = BUPTCampus_Dataset(
            data_root=opt.bupt_root,
            mode=mode,
            sample=sample_mode,
            seq_len=opt.sequence_length,
            transform=transform,
            transform2=transform2,
            transform3=transform3,
            random_flip=opt.random_flip,
            fake=opt.fake,
        )
    else:
        raise RuntimeError(f'Dataset {opt.dataset} is not supported for now.')

    if show:
        dataset.show_information()

    if mode == 'train':
        if opt.train_sampler is None:
            dataloader = DataLoader(
                dataset,
                batch_size=opt.bupt_train_batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=opt.num_workers,
            )
        elif opt.train_sampler == 'RandomIdentitySampler':
            sampler = RandomIdentitySampler(
                dataset,
                np=opt.bupt_train_batch_size // (opt.cameras_per_identity * opt.tracklets_per_camera),
                nc=opt.cameras_per_identity,
                nt=opt.tracklets_per_camera,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=opt.bupt_train_batch_size,
                sampler=sampler,
                drop_last=True,
                num_workers=opt.num_workers,
            )
    elif mode == 'auxiliary':
        if opt.auxiliary_sampler is None:
            dataloader = DataLoader(
                dataset,
                batch_size=opt.bupt_train_batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=opt.num_workers,
            )
        elif opt.auxiliary_sampler == 'RandomCameraSampler':
            sampler = RandomCameraSampler(
                dataset,
                np=opt.bupt_train_batch_size // (opt.auxiliary_sampler_nc * opt.auxiliary_sampler_nt),
                nc=opt.auxiliary_sampler_nc,
                nt=opt.auxiliary_sampler_nt,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=opt.bupt_train_batch_size,
                sampler=sampler,
                drop_last=False,
                num_workers=opt.num_workers,
            )
    else:
        if opt.test_sampler is None:
            dataloader = DataLoader(
                dataset,
                batch_size=opt.bupt_test_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=opt.num_workers,
            )
        elif opt.test_sampler == 'ConsistentModalitySampler':
            sampler = ConsistentModalitySampler(
                dataset,
                batch_size=opt.bupt_test_batch_size,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=opt.bupt_test_batch_size,
                sampler=sampler,
                drop_last=False,
                num_workers=opt.num_workers,
            )
    return dataloader, dataset.get_class_num(),dataset

class BUPTCampus_Dataset(Dataset):
    def __init__(self, data_root, mode, sample, seq_len, transform,transform2=None,transform3=None, random_flip=False, fake=False):
        """
        :param data_root:
        :param mode: 'train', 'query', 'gallery'
        :param sample: 'dense', 'uniform'
        :param seq_len:
        :param transform:
        """
        assert mode in ('train', 'query', 'gallery', 'auxiliary')
        self.mode = mode
        self.sample = sample
        self.seq_len = seq_len
        self.data_root = data_root
        self.img_root = join(data_root,'DATA')
        self.transform = transform
        self.transform2 = transform2
        self.transform3 = transform3
        self.random_flip = random_flip
        self.fake = fake

        self.data_info = self.parse_data()

        self.data_paths = json.load(open(join(data_root, 'data_paths.json')))

        self.pid2label = {pid: label for label, pid in enumerate(self.pids)}

        if mode == 'train':
            self.train_ir, self.train_rgb = self._process_train(self.data_info)

    def count_tracklets_per_identity(self,tracklets):
        """
        tracklets: [(data_paths, obj_id, camera), ...]
        """
        id2count = defaultdict(int)
        for _, obj_id, _ in tracklets:
            id2count[obj_id] += 1  # 统计每个身份有多少个序列

        distribution = Counter(id2count.values())

        return id2count, distribution

    def _process_train(self, data_info):
        tracklets_ir = []
        tracklets_rgb = []
        for i, data in enumerate(data_info):
            obj_id, modality, camera, tracklet_id = data

            data_paths_ir = self.data_paths[obj_id]['IR'][camera][tracklet_id]
            data_paths_ir = [osp.join(self.img_root, img_name) for img_name in data_paths_ir]
            tracklets_ir.append((data_paths_ir, obj_id, camera))

            data_paths_rgb = self.data_paths[obj_id]['RGB'][camera][tracklet_id]
            data_paths_rgb = [osp.join(self.img_root, img_name) for img_name in data_paths_rgb]
            tracklets_rgb.append((data_paths_rgb, obj_id, camera))

        tracklets_ir, dbg_ir = split_tracklets_balanced(
            tracklets_ir,
            clip_len=50,
            max_clips=3,
            mid_clips=3,
            short_views=3,   # 短序列尽量 2 个 view（你也可以设 3）
            pad_mode="cycle"
        )

        tracklets_rgb, dbg_rgb = split_tracklets_balanced(
            tracklets_rgb,
            clip_len=50,
            max_clips=3,
            mid_clips=3,
            short_views=3,
            pad_mode="cycle"
        )

        tracklets_ir = relabel(tracklets_ir)
        tracklets_rgb = relabel(tracklets_rgb)
        return tracklets_ir, tracklets_rgb

    def parse_data(self):
        if self.mode == 'train':
            data_info = self._parse_data(join(self.data_root,'train.txt'))
        elif self.mode == 'query':
            data_info = self._parse_data(join(self.data_root,'query.txt'))
        elif self.mode == 'gallery':
            data_info = self._parse_data(join(self.data_root,'gallery.txt'))
        elif self.mode == 'auxiliary':
            data_info = self._parse_data(join(self.data_root,'train_auxiliary.txt'))
        return data_info

    def _parse_data(self, path):
        data_info, pids = [], []
        path = join(self.data_root, path)
        with open(path) as f:
            for line in f.readlines():
                obj_id, modality, camera, tracklet_id = line.strip().split(' ')
                data_info.append((obj_id, modality, camera, tracklet_id))
                pids.append(obj_id)
        self.pids = sorted(set(pids))
        return data_info

    def get_class_num(self):
        return len(self.pids)

    def fast_iteration(self):
        iteration = [
            [self.pid2label[obj_id], MODALITY[modality], CAMERA[camera]]
            for (obj_id, modality, camera, tracklet_id) in self.data_info
        ]
        return iter(iteration)

    def __getitem__(self, index):
        obj_id, modality, camera, tracklet_id = self.data_info[index]

        if self.mode in ('train', 'auxiliary'):
            """
            Please note that every sample has two modalities while training,
            which means that the final batch size is equal to `2*opt.train_bs`
            """
            if modality == 'RGB/IR':
                data_paths_ir = self.data_paths[obj_id]['IR'][camera][tracklet_id]
                data_paths_rgb = self.data_paths[obj_id]['RGB'][camera][tracklet_id]
                if self.fake:
                    data_paths_rgb = [x.replace('/RGB/', '/FakeIR/') for x in data_paths_rgb]
                tra_len = len(data_paths_ir)
            else:
                raise RuntimeError('Only modality RGB/IR is supported for training.')

            if self.sample == 'random':
                replace = tra_len < self.seq_len
                frame_idx = sorted(choice(range(tra_len), size=self.seq_len, replace=replace))
            elif self.sample == 'restricted_random':
                frame_idx = list()
                if tra_len >= self.seq_len:
                    step = tra_len / self.seq_len
                    tra_idx = list(range(tra_len))
                else:
                    step = 1
                    tra_idx = [0] * (self.seq_len - tra_len) + list(range(tra_len))
                for i in range(self.seq_len):
                    idx = tra_idx[int(i*step): int((i+1)*step)]
                    frame_idx += random.sample(idx, 1)
            else:
                raise RuntimeError(f'Wrong sampling method {self.sample}.')

            images_ir = torch.cat(
                [self.transform3(
                    Image.open(join(self.img_root, data_paths_ir[idx])))
                    for idx in frame_idx],
                dim=0
            )  # [T,C,H,W]
            idx = random.randint(0, 1)
            if idx==0:
                images_rgb = torch.cat(
                    [self.transform(
                        Image.open(join(self.img_root, data_paths_rgb[idx])))
                        for idx in frame_idx],
                    dim=0
                )  # [T,C,H,W]
            elif idx==1:
                images_rgb = torch.cat(
                    [self.transform2(
                        Image.open(join(self.img_root, data_paths_rgb[idx])))
                        for idx in frame_idx],
                    dim=0
                )  # [T,C,H,W]

            if self.random_flip and torch.rand(1) < 0.5:
                images_ir = images_ir.flip(-1)
                images_rgb = images_rgb.flip(-1)

            label = self.pid2label[obj_id]

            return images_rgb, images_ir, label, label

        else:
            data_paths = self.data_paths[obj_id][modality][camera][tracklet_id]
            if self.fake:
                data_paths = [x.replace('/RGB/', '/FakeIR/') for x in data_paths]
            tra_len = len(data_paths)

            if self.sample == 'dense':
                '''
                Sample all frames for a tracklet.
                Only batch_size=1 is supported for this mode.
                '''
                frame_idx = range(tra_len)
            elif self.sample == 'uniform':
                '''
                Uniform sampling frames for a tracklet.
                '''
                frame_idx = np.linspace(0, tra_len, self.seq_len, endpoint=False, dtype=int)
            elif self.sample == 'first_half':
                frame_idx = np.linspace(0, tra_len//2, self.seq_len, endpoint=False, dtype=int)
            elif self.sample == 'second_half':
                frame_idx = np.linspace(tra_len//2, tra_len, self.seq_len, endpoint=False, dtype=int)
            else:
                raise RuntimeError(f'Wrong sampling method {self.sample}.')

            images = torch.cat(
                [self.transform(Image.open(join(self.img_root, data_paths[idx])))
                 for idx in frame_idx],
                dim=0
            )  # [T,C,H,W]

            return images, int(obj_id), CAMERA[camera],MODALITY[modality]

    def __len__(self):
        return len(self.data_info)

    def show_information(self):
        if self.mode in ('train', 'auxiliary'):
            factor = 2
        else:
            factor = 1
        print(
            f"===> MCPRL-ReID Dataset ({self.mode}) <===\n"
            f"Number of identities: {len(self.pids)}\n"
            f"Number of samples   : {len(self.data_info) * factor}"
        )
