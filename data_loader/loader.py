
import torchvision.transforms as transforms
from data_loader.dataset import SYSUData, RegDBData, TestData, process_query_sysu, process_gallery_sysu,\
    process_test_regdb
from data_loader.processing import ChannelRandomErasing, ChannelAdapGray, ChannelExchange
from data_loader.preprocessor import VideoPreprocessor
from data_loader.sampler import GenIdx, IdentitySampler,RandomMultipleGallerySampler,SingleModalIdentitySampler,GenIdx_Hard

import torch.utils.data as data
from data_loader.vcm_data_manager import VCM
from data_loader.vcm_dataloader import *
from data_loader.bupt_dataloader import get_dataloader
from torch.utils.data import DataLoader

class Loader:

    def __init__(self, config):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.transform_test = transforms.Compose([

            transforms.Resize((config.image_height, config.image_width)),
            transforms.ToTensor(),
            normalize])

        self.transform_color1_vcm  = transforms.Compose([
            transforms.Resize((config.image_height, config.image_width)),
            transforms.Pad(10),
            transforms.RandomCrop((config.image_height, config.image_width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5)])

        self.transform_color2_vcm  = transforms.Compose([
            transforms.Resize((config.image_height, config.image_width)),
            transforms.Pad(10),
            transforms.RandomCrop((config.image_height, config.image_width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5),

            ]
            )

        self.transform_thermal_vcm  = transforms.Compose([
            transforms.Resize((config.image_height, config.image_width)),
            transforms.Pad(10),
            transforms.RandomCrop((config.image_height, config.image_width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5),
            ChannelAdapGray(probability=0.5)])

        self.dataset = config.dataset
        self.image_width = config.image_width
        self.image_height = config.image_height
        self.instances_per_identity = config.instances_per_identity
        self.batch_size = config.batch_size
        self.ciw_batch_size = config.ciw_batch_size
        self.num_workers = config.num_workers
        self.sequence_length = config.sequence_length
        self.vcm_test_batch_size = config.vcm_test_batch_size
        self.config = config
        self._loader()

    def _loader(self):
        if self.dataset == 'vcm':
            samples = VCM(self.config)
            self.color_pos, self.thermal_pos = GenIdx(samples.rgb_label, samples.ir_label)
            self.samples = samples

            self.query_loader = data.DataLoader(
                VideoDataset_test(samples.query, seq_len=self.sequence_length, sample='video_test', transform=self.transform_test),
                batch_size=self.vcm_test_batch_size, shuffle=False, num_workers=self.num_workers)

            self.gallery_loader = data.DataLoader(
                VideoDataset_test(samples.gallery, seq_len=self.sequence_length, sample='video_test', transform=self.transform_test),
                batch_size=self.vcm_test_batch_size, shuffle=False, num_workers=self.num_workers)

            self.query_loader_1 = data.DataLoader(
                VideoDataset_test(samples.query_1, seq_len=self.sequence_length, sample='video_test', transform=self.transform_test),
                batch_size=self.vcm_test_batch_size, shuffle=False, num_workers=self.num_workers)

            self.gallery_loader_1 = data.DataLoader(
                VideoDataset_test(samples.gallery_1, seq_len=self.sequence_length, sample='video_test', transform=self.transform_test),
                batch_size=self.vcm_test_batch_size, shuffle=False, num_workers=self.num_workers)

            self.rgb_cluster_loader = data.DataLoader(
                VideoDataset_test(samples.train_rgb, seq_len=self.sequence_length, sample='video_cluster', transform=self.transform_test),
                batch_size=self.vcm_test_batch_size, shuffle=False, num_workers=self.num_workers)

            self.ir_cluster_loader =  data.DataLoader(
                VideoDataset_test(samples.train_ir, seq_len=self.sequence_length, sample='video_cluster', transform=self.transform_test),
                batch_size=self.vcm_test_batch_size, shuffle=False, num_workers=self.num_workers)

            self.nquery_1 = samples.num_query_tracklets_1
            self.ngall_1 = samples.num_gallery_tracklets_1

            self.n_class = samples.num_train_pids
            self.nquery = samples.num_query_tracklets
            self.ngall = samples.num_gallery_tracklets
        elif self.dataset == 'bupt':
            self.query_loader, _,_ = get_dataloader(self.config, 'query',transform=self.transform_test, show=False)
            self.gallery_loader, _,_ = get_dataloader(self.config, 'gallery', transform=self.transform_test, show=False)
            self.dataloader_train, self.class_num,self.datatset_bupt = get_dataloader(self.config, 'train',transform=self.transform_color1_vcm,transform2=self.transform_color2_vcm,transform3=self.transform_thermal_vcm, show=True)
            self.train_iter = IterLoader(self.dataloader_train)

            self.samples = self.datatset_bupt

            self.rgb_cluster_loader = data.DataLoader(
                VideoDataset_test(self.datatset_bupt.train_rgb, seq_len=self.sequence_length, sample='video_cluster', transform=self.transform_test),
                batch_size=self.vcm_test_batch_size, shuffle=False, num_workers=self.num_workers)

            self.ir_cluster_loader =  data.DataLoader(
                VideoDataset_test(self.datatset_bupt.train_ir, seq_len=self.sequence_length, sample='video_cluster', transform=self.transform_test),
                batch_size=self.vcm_test_batch_size, shuffle=False, num_workers=self.num_workers)
            print('bupt loader finished!')

    def get_cluster_loader(self,pseudo_labeled_dataset_rgb,pseudo_labeled_dataset_ir,rgb_labels,ir_labels):
        color_pos, thermal_pos = GenIdx(rgb_labels, ir_labels)
        rgb_sampler = SingleModalIdentitySampler(rgb_labels,color_pos,self.instances_per_identity,self.batch_size)
        ir_sampler = SingleModalIdentitySampler(ir_labels,thermal_pos,self.instances_per_identity,self.batch_size)
        rgb_index = rgb_sampler.index1
        ir_index2 = ir_sampler.index1
        loader_batch = self.batch_size * self.instances_per_identity

        train_loader_rgb = data.DataLoader(
            VideoDataset_train_rgb(pseudo_labeled_dataset_rgb, seq_len=self.sequence_length, sample='video_train',
                                           transform1=self.transform_color1_vcm, transform2=self.transform_color2_vcm,
                                           transform3=self.transform_thermal_vcm),
            sampler=rgb_sampler,batch_size=loader_batch, num_workers=self.num_workers,
            drop_last=True
        )
        self.cluster_iter_rgb = IterLoader(train_loader_rgb)

        train_loader_ir = data.DataLoader(
            VideoDataset_train_ir(pseudo_labeled_dataset_ir, seq_len=self.sequence_length, sample='video_train',
                                          transform1=self.transform_color1_vcm, transform2=self.transform_color2_vcm,
                                          transform3=self.transform_thermal_vcm),
            sampler=ir_sampler,batch_size=loader_batch, num_workers=self.num_workers,
            drop_last=True
        )

        self.cluster_iter_ir = IterLoader(train_loader_ir)

        return self.cluster_iter_rgb, self.cluster_iter_ir

    def get_unified_modal_loader(self,pseudo_labeled_dataset_rgb,pseudo_labeled_dataset_ir,rgb_labels,ir_labels):

        if self.config.dataset=='vcm':
            color_pos, thermal_pos = GenIdx(rgb_labels, ir_labels)

            print('==> Preparing Data Loader...')
            sampler = IdentitySampler(ir_labels, rgb_labels, color_pos, thermal_pos, self.instances_per_identity, self.batch_size)
            index1 = sampler.index1
            index2 = sampler.index2

            loader_batch = self.batch_size * self.instances_per_identity

            train_loader = data.DataLoader(
                VideoDataset_train(pseudo_labeled_dataset_ir, pseudo_labeled_dataset_rgb, seq_len=self.sequence_length, sample='video_train',
                                   transform1=self.transform_color1_vcm, transform2=self.transform_color2_vcm,
                                   transform3=self.transform_thermal_vcm, index1=index1, index2=index2),
                sampler=sampler,
                batch_size=loader_batch, num_workers=self.num_workers,
                drop_last=True,
            )

            train_iter = IterLoader(train_loader)

            return train_iter
        else:
            color_pos, thermal_pos = GenIdx(rgb_labels, ir_labels)

            print('==> Preparing Data Loader...')
            sampler = IdentitySampler(ir_labels, rgb_labels, color_pos, thermal_pos, self.instances_per_identity, self.batch_size)
            index1 = sampler.index1
            index2 = sampler.index2

            loader_batch = self.batch_size * self.instances_per_identity

            train_loader = data.DataLoader(
                VideoDataset_train(pseudo_labeled_dataset_ir, pseudo_labeled_dataset_rgb, seq_len=self.sequence_length, sample='video_train',
                                   transform1=self.transform_color1_vcm, transform2=self.transform_color2_vcm,
                                   transform3=self.transform_thermal_vcm, index1=index1, index2=index2),
                sampler=sampler,
                batch_size=loader_batch, num_workers=self.num_workers,
                drop_last=True,
            )

            train_iter = IterLoader(train_loader)

            return train_iter

    def get_hard_modal_loader(self,pseudo_labeled_dataset_rgb,pseudo_labeled_dataset_ir,rgb_labels,ir_labels):

        if self.config.dataset=='vcm':
            color_pos, thermal_pos = GenIdx_Hard(rgb_labels, ir_labels)

            print('==> Preparing Data Loader...')
            sampler = IdentitySampler(ir_labels, rgb_labels, color_pos, thermal_pos, self.instances_per_identity, self.batch_size)
            index1 = sampler.index1
            index2 = sampler.index2

            loader_batch = self.batch_size * self.instances_per_identity

            train_loader = data.DataLoader(
                VideoDataset_train(pseudo_labeled_dataset_ir, pseudo_labeled_dataset_rgb, seq_len=self.sequence_length, sample='video_train',
                                   transform1=self.transform_color1_vcm, transform2=self.transform_color2_vcm,
                                   transform3=self.transform_thermal_vcm, index1=index1, index2=index2),
                sampler=sampler,
                batch_size=loader_batch, num_workers=self.num_workers,
                drop_last=True,
            )

            train_iter = IterLoader(train_loader)

            return train_iter
        else:
            color_pos, thermal_pos = GenIdx_Hard(rgb_labels, ir_labels)

            print('==> Preparing Data Loader...')
            sampler = IdentitySampler(ir_labels, rgb_labels, color_pos, thermal_pos, self.instances_per_identity, self.batch_size)
            index1 = sampler.index1
            index2 = sampler.index2

            loader_batch = self.batch_size * self.instances_per_identity

            train_loader = data.DataLoader(
                VideoDataset_train(pseudo_labeled_dataset_ir, pseudo_labeled_dataset_rgb, seq_len=self.sequence_length, sample='video_train',
                                   transform1=self.transform_color1_vcm, transform2=self.transform_color2_vcm,
                                   transform3=self.transform_thermal_vcm, index1=index1, index2=index2),
                sampler=sampler,
                batch_size=loader_batch, num_workers=self.num_workers,
                drop_last=True,
            )

            train_iter = IterLoader(train_loader)

            return train_iter

    def get_noise_loader(self,pseudo_labeled_dataset_rgb,pseudo_labeled_dataset_ir):
        loader_batch = self.batch_size * self.instances_per_identity
        train_loader_rgb = data.DataLoader(
            VideoDataset_train_rgb(pseudo_labeled_dataset_rgb, seq_len=self.sequence_length, sample='video_train',
                                           transform1=self.transform_color1_vcm, transform2=self.transform_color2_vcm,
                                           transform3=self.transform_thermal_vcm),batch_size=loader_batch, num_workers=self.num_workers,
            drop_last=True
        )
        self.noise_iter_rgb = IterLoader(train_loader_rgb)

        train_loader_ir = data.DataLoader(
            VideoDataset_train_ir(pseudo_labeled_dataset_ir, seq_len=self.sequence_length, sample='video_train',
                                          transform1=self.transform_color1_vcm, transform2=self.transform_color2_vcm,
                                          transform3=self.transform_thermal_vcm),batch_size=loader_batch, num_workers=self.num_workers,
            drop_last=True
        )

        self.noise_iter_ir = IterLoader(train_loader_ir)

        return self.noise_iter_rgb, self.noise_iter_ir

    def get_train_loader_ir(self, dataset, batch_size, workers,
                            num_instances, iters, trainset=None, no_cam=False, train_transformer=None):

        train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
        rmgs_flag = num_instances > 0
        if rmgs_flag:
            if no_cam:
                sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
            else:
                sampler = RandomMultipleGallerySampler(train_set, num_instances)
        else:
            sampler = None
        train_loader = IterLoader(
            data.DataLoader(VideoDataset_train_ir(train_set, root=dataset.images_dir, transform=train_transformer),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

        return train_loader

    def get_train_loader(self):
        if self.config.dataset=='vcm':
            print('==> Preparing Data Loader...')
            sampler = IdentitySampler(self.samples.ir_label, self.samples.rgb_label, self.color_pos, self.thermal_pos, self.instances_per_identity, self.batch_size)
            index1 = sampler.index1
            index2 = sampler.index2

            loader_batch = self.batch_size * self.instances_per_identity

            train_loader = data.DataLoader(
                VideoDataset_train(self.samples.train_ir, self.samples.train_rgb, seq_len=self.sequence_length, sample='video_train',
                                   transform1=self.transform_color1_vcm, transform2=self.transform_color2_vcm,
                                   transform3=self.transform_thermal_vcm, index1=index1, index2=index2),
                sampler=sampler,
                batch_size=loader_batch, num_workers=self.num_workers,
                drop_last=True,
            )

            train_iter = IterLoader(train_loader)

            train_loader_rgb = data.DataLoader(
                VisibleInterventionDataset(self.samples.train_rgb,self.samples.train_ir,  seq_len=self.sequence_length, sample='video_train',
                                   transform1=self.transform_color1_vcm, transform2=self.transform_color2_vcm,
                                   transform3=self.transform_thermal_vcm, index1=index1, index2=index2),
                batch_size=self.ciw_batch_size, num_workers=self.num_workers,
                drop_last=True,shuffle=True
            )
            train_iter_rgb = IterLoader(train_loader_rgb)

            train_loader_ir = data.DataLoader(
                InfraredInterventionDataset(self.samples.train_ir,self.samples.train_rgb, seq_len=self.sequence_length, sample='video_train',
                                   transform1=self.transform_color1_vcm, transform2=self.transform_color2_vcm,
                                   transform3=self.transform_thermal_vcm, index1=index1, index2=index2),
                batch_size=self.ciw_batch_size, num_workers=self.num_workers,
                drop_last=True,shuffle=True
            )

            train_iter_ir = IterLoader(train_loader_ir)

            return train_iter,train_iter_rgb,train_iter_ir
        else:
            print('==> Preparing Data Loader...')
            loader_batch = self.ciw_batch_size

            train_loader_rgb = data.DataLoader(
                VisibleInterventionDataset(self.datatset_bupt.train_rgb, self.datatset_bupt.train_ir, seq_len=self.sequence_length,
                                               sample='video_train',
                                               transform1=self.transform_color1_vcm,
                                               transform2=self.transform_color2_vcm,
                                               transform3=self.transform_thermal_vcm),
                batch_size=loader_batch, num_workers=self.num_workers,
                drop_last=True, shuffle=True
            )
            train_iter_rgb = IterLoader(train_loader_rgb)

            train_loader_ir = data.DataLoader(
                InfraredInterventionDataset(self.datatset_bupt.train_ir, self.datatset_bupt.train_rgb, seq_len=self.sequence_length,
                                              sample='video_train',
                                              transform1=self.transform_color1_vcm,
                                              transform2=self.transform_color2_vcm,
                                              transform3=self.transform_thermal_vcm),
                batch_size=loader_batch, num_workers=self.num_workers,
                drop_last=True, shuffle=True
            )

            train_iter_ir = IterLoader(train_loader_ir)

            return train_iter_rgb, train_iter_ir

    def get_test_loader(self,dataset, height, width, batch_size, workers, testset=None, test_transformer=None):

        test_loader = DataLoader(
            VideoPreprocessor(testset, root=dataset.images_dir, transform=test_transformer),
            batch_size=batch_size, num_workers=workers,
            shuffle=False, pin_memory=True)

        return test_loader

class IterLoader:

    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(self.loader)

    def next_one(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)
