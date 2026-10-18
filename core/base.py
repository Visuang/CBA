import os
import torch
import torch.nn as nn
from network import  Classifier,Model
from tools import os_walk, TripletLoss_WRT,SupConLoss,InterventionInfoNCELoss
from network.scheduler import CosineLRScheduler
from network.cm import ClusterMemory
import torch.nn.functional as F

class Base:
    def __init__(self, config):
        self.config = config
        self.num_train_ids = config.num_train_ids
        self.output_dir = config.output_dir
        self.save_model_path = os.path.join(self.output_dir, 'models/')
        self.save_logs_path = os.path.join(self.output_dir, 'logs/')

        self.learning_rate = config.learning_rate
        self.alignment_learning_rate = config.alignment_learning_rate
        self.lr_multiplier = config.lr_multiplier
        self.weight_decay = config.weight_decay
        self.steps_per_epoch = config.steps_per_epoch

        self.image_height = config.image_height
        self.image_width = config.image_width
        self.sequence_length = config.sequence_length
        self.pgur_ir_weight = config.pgur_ir_weight
        self.pgur_rgb_weight = config.pgur_rgb_weight
        self.ttb_weight = config.ttb_weight
        self.ics_weight = config.ics_weight
        self.mpb_weight = config.mpb_weight

        self.temporal_prompt_start_layer = config.temporal_prompt_start_layer
        self.cluster_ir = None
        self.cluster_rgb = None
        self.alternate = config.alternate
        self.gpu = config.gpu

        self._init_device()
        self._init_model()
        self._init_creiteron()
        self._init_optimizer()

        self.temp = 0.05
        self.momentum = 0.1

    def _init_device(self):
        self.device = torch.device(f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")
        print(f"device = {self.device}")

    def _init_model(self):
        self.model = Model(self.device,self.config,self.num_train_ids, self.image_height, self.image_width)
        self.model = self.model.to(self.device)

    def _get_ClusterMemoery(self,cluster_features_ir,cluster_features_rgb,num_cluster_ir,num_cluster_rgb):

        self.memory_ir = ClusterMemory(1536, num_cluster_ir, temp=self.temp,
                               momentum=self.momentum, use_hard=False).to(self.device)
        self.memory_rgb = ClusterMemory(1536, num_cluster_rgb, temp=self.temp,
                               momentum=self.momentum, use_hard=False).to(self.device)
        self.memory_ir.features = F.normalize(cluster_features_ir, dim=1).to(self.device)
        self.memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).to(self.device)

    def _init_creiteron(self):
        self.pid_creiteron = nn.CrossEntropyLoss()
        self.tri_creiteron = TripletLoss_WRT()
        self.con_creiteron = SupConLoss(self.device)
        self.intervention_creiteron = InterventionInfoNCELoss()

    def _init_optimizer(self):
        params = []
        keys = []
        for key, value in self.model.named_parameters():
            if "text_encoder" in key:
                value.requires_grad_(False)
                continue
            if not value.requires_grad:
                continue
            lr = self.learning_rate
            weight_decay = self.weight_decay
            if "bias" in key:
                lr = self.learning_rate * self.lr_multiplier
                weight_decay = self.weight_decay

            if "classifier" in key or "arcface" in key:
                lr = self.learning_rate * self.lr_multiplier

            if "prompt_learner" in key:
                lr = self.learning_rate * self.lr_multiplier

            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]

        self.model_optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        lr_min = self.learning_rate*0.01
        warmup_lr_init = self.learning_rate*0.01
        warmup_t = 3
        self.model_lr_scheduler = CosineLRScheduler(
            self.model_optimizer,
            t_initial=24,
            lr_min=lr_min,
            t_mul= 1.,
            decay_rate=1.0,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_t,
            cycle_limit=1,
            t_in_epochs=True,
            noise_range_t=None,
            noise_pct= 0.67,
            noise_std= 1.,
            noise_seed=42,
        )

    def set_lr(self,logger):
        for key, param_group in zip(self.model.named_parameters(), self.model_optimizer.param_groups):
            name, _ = key

            lr = self.alignment_learning_rate
            weight_decay = self.weight_decay

            if "bias" in name:
                lr = self.alignment_learning_rate * self.lr_multiplier
                weight_decay = self.weight_decay

            if "classifier" in name or "arcface" in name:
                lr = self.alignment_learning_rate * self.lr_multiplier

            if "prompt_learner" in name:
                lr = self.alignment_learning_rate * self.lr_multiplier

            param_group['lr'] = lr
            param_group['weight_decay'] = weight_decay

        self.model_lr_scheduler = None
        logger(f"Switched to fixed LR with new base: {self.alignment_learning_rate}")

    def save_model(self, save_epoch,is_best,type='best'):
        if type == 'seq':
            name = 'best_seq.pth'
        elif type == 'intra':
            name = 'best_intra.pth'
        elif type == 'best':
            name = 'best_model.pth'

        if is_best:
            model_file_path = os.path.join(self.save_model_path, name)
            torch.save(self.model.state_dict(), model_file_path)

    def resume_model(self, resume_path):
        model_path = os.path.join(resume_path, 'best_model.pth')
        self.model.load_state_dict(torch.load(model_path), strict=False)
        print('Successfully resume shared_model from {}'.format(model_path))

    def resume_the_model(self, resume_path):

        self.model.load_state_dict(torch.load(resume_path), strict=False)
        print('Successfully resume shared_model from {}'.format(resume_path))

    def export_model(self, model_name="model",logger=None):
        """
        保存模型

        Args:
            model_name (str): 模型文件名（不带扩展名）
        """
        model_file_path = os.path.join(
            self.save_model_path, f"{model_name}.pth"
        )
        torch.save(self.model.state_dict(), model_file_path)
        msg = f"模型已保存到: {model_file_path}"
        if logger is not None:
            logger(msg)
        else:
            print(msg)

    def set_train(self):
        self.model = self.model.train()
        self.training = True

    def set_eval(self):
        self.model = self.model.eval()
        self.training = False
