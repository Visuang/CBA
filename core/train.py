import torch
from tools import MultiItemAverageMeter
from core.video_train import train_video,train_ciw,train_intra_modality,train_pgur
from torch.cuda import amp

def train(base, loaders, config,scaler,mode='sequence',current_epoch=0,logger=None):

    base.set_train()
    meter = MultiItemAverageMeter()
    if config.dataset=='vcm':
        loader,iter_rgb,iter_ir = loaders.get_train_loader()
    elif config.dataset=='bupt':
        iter_rgb,iter_ir = loaders.get_train_loader()

    if mode == 'sequence':
        meter = train_ciw(iter_rgb,iter_ir, base, meter, scaler,current_epoch)
    elif mode == 'video':
        meter = train_video(loader,base,meter,scaler)
    elif mode == 'intra_modality':
        meter = train_intra_modality(loaders, base, meter, scaler,current_epoch,logger)
    elif mode == 'inter_modality':
        meter = train_pgur(loaders, base, meter, scaler,current_epoch,logger)

    return meter.get_val(), meter.get_str()
