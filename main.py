from datetime import datetime
import os

import ast
import torch
import random
import argparse
import numpy as np
from torch.cuda import amp
from data_loader.loader import Loader
from core import Base, train, test,test_vcm,test_bupt,get_cluster_results,generate_cluster_features
from tools import make_dirs, Logger, os_walk, time_now
from core import extract_features
import warnings
warnings.filterwarnings("ignore")

import setproctitle

setproctitle.setproctitle("cba")

best_mAP = 0
best_rank1 = 0
def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_and_save(model, loaders, config, current_epoch, best_rank1, logger,type='seq'):
    """
    Evaluate model on specific dataset and save checkpoint if best.

    Args:
        model: 模型对象，必须实现 save_model(epoch, is_best) 方法
        loaders: 数据加载器
        config: 配置对象，需包含 dataset, eval_epoch
        current_epoch: 当前 epoch
        best_rank1: 历史最好 Rank-1
        logger: 日志函数
    Returns:
        best_rank1 (float): 更新后的 best rank-1
    """
    if (current_epoch + 1) % config.eval_interval != 0:
        return best_rank1  # 不在评估周期内，直接返回

    if config.dataset == 'bupt':
        cmc, mAP, eval_str = test_bupt(model, loaders, config)
        is_best_rank = (cmc[1][0] >= best_rank1)
        best_rank1 = max(cmc[1][0], best_rank1)
        model.save_model(current_epoch, is_best_rank,type)
        logger('Time: {}; Test on Dataset: {}, \n task: {}, \n task: {}'.format(
            time_now(), config.dataset, eval_str[0], eval_str[1]
        ))

    elif config.dataset == 'vcm':
        cmc, mAP, cmc2, mAP2, eval_str_t2v, eval_str_v2t = test_vcm(model, loaders)
        is_best_rank = (cmc[0] >= best_rank1)
        best_rank1 = max(cmc[0], best_rank1)
        model.save_model(current_epoch, is_best_rank,type)
        logger('Time: {}; Test on Dataset: {}, \n task: {}, \n task: {}'.format(
            time_now(), config.dataset, eval_str_t2v, eval_str_v2t
        ))

    return best_rank1

def evaluate(model, loaders, config, logger):
    if config.dataset == 'bupt':
        cmc, mAP, eval_str = test_bupt(model, loaders, config)
        logger('Time: {}; Test on Dataset: {}, \n task: {}, \n task: {}'.format(time_now(), config.dataset,
                                                                                eval_str[0], eval_str[1]))
    elif config.dataset == 'vcm':
        cmc, mAP, cmc2, mAP2, eval_str_t2v, eval_str_v2t = test_vcm(model, loaders)
        logger('Time: {}; Test on Dataset: {}, \n task: {}, \n task: {}'.format(time_now(), config.dataset,
                                                                                eval_str_t2v, eval_str_v2t))

def main(config):
    global best_mAP
    global best_rank1

    loaders = Loader(config)
    model = Base(config)

    make_dirs(model.output_dir)
    make_dirs(model.save_model_path)
    make_dirs(model.save_logs_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式: 20251006_132455
    log_name = f"log_{config.dataset}_{config.mode}_gpu{config.gpu}_{timestamp}.log"

    logger = Logger(os.path.join(os.path.join(config.output_dir, 'logs/'), log_name))
    logger('\n' * 3)
    logger(config)
    logger(log_name)

    if config.mode == 'train':
        start_train_epoch = 0

        scaler = amp.GradScaler()

        for current_epoch in range(start_train_epoch, config.ciw_epochs):
            model.model_lr_scheduler.step(current_epoch)
            _, result = train(model, loaders, config, scaler)
            logger('Time: {}; Epoch: {}; {}; {}'.format(time_now(), current_epoch, result,model.model_optimizer.state_dict()['param_groups'][0]['lr']))

        model.set_lr(logger)

        evaluate(model, loaders, config, logger)

        for current_epoch in range(start_train_epoch, config.intra_epochs):
            _, result = train(model, loaders, config, scaler,mode = 'intra_modality',current_epoch=current_epoch,logger=logger)
            logger('Time: {}; Epoch: {}; {}; {}'.format(time_now(), current_epoch, result,
                                                        model.model_optimizer.state_dict()['param_groups'][0]['lr']))

        model.set_lr(logger)

        evaluate(model, loaders, config, logger)

        for current_epoch in range(start_train_epoch, config.pgur_epochs):
            _, result = train(model, loaders, config, scaler, mode='inter_modality',
                              current_epoch=current_epoch, logger=logger)
            logger('Time: {}; Epoch: {}; {}; {}'.format(time_now(), current_epoch, result,
                                                        model.model_optimizer.state_dict()['param_groups'][0]['lr']))

            best_rank1 = evaluate_and_save(model, loaders, config, current_epoch, best_rank1, logger,type='best')

    elif config.mode == 'test':
        model.resume_model(config.checkpoint_dir)
        if config.dataset == 'bupt':
            cmc, mAP, eval_str = test_bupt(model, loaders, config)
            logger('Time: {}; Test on Dataset: {}, \n task: {}, \n task: {}'.format(time_now(), config.dataset,
                                                                                    eval_str[0], eval_str[1]))
        elif config.dataset == 'vcm':
            cmc, mAP, cmc2, mAP2, eval_str_t2v, eval_str_v2t = test_vcm(model, loaders)
            logger('Time: {}; Test on Dataset: {}, \n task: {}, \n task: {}'.format(time_now(), config.dataset,
                                                                                    eval_str_t2v, eval_str_v2t))

def build_parser():
    parser = argparse.ArgumentParser(
        description="Causal Bootstrapped Alignment (CBA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--gpu', dest='gpu', type=int, default=0, help="GPU id to use")
    parser.add_argument('--cuda', dest='cuda', type=str, default='cuda')
    parser.add_argument('--mode', dest='mode', type=str, default='train', choices=('train', 'test'), help='Execution mode')
    parser.add_argument('--dataset', dest='dataset', default='vcm', choices=('vcm', 'bupt'), help='Dataset')
    parser.add_argument('--vcm-root', dest='vcm_root', type=str, default=os.environ.get('CBA_VCM_ROOT', '/mnt/share/ubuntu/data/HITSZ-VCM/'))
    parser.add_argument('--bupt-root', dest='bupt_root', type=str, default=os.environ.get('CBA_BUPT_ROOT', '/mnt/share/ubuntu/data/BUPTCampus/'))
    parser.add_argument('--ciw-batch-size', dest='ciw_batch_size', default=8, type=int, metavar='B', help='training batch size')
    parser.add_argument('--batch-size', dest='batch_size', default=4, type=int, metavar='B', help='training batch size')
    parser.add_argument('--instances-per-identity', dest='instances_per_identity', default=4, type=int,
                        help='Number of instances sampled per identity')
    parser.add_argument('--image-width', dest='image_width', default=144, type=int, metavar='imgw', help='Input image width')
    parser.add_argument('--image-height', dest='image_height', default=288, type=int, metavar='imgh', help='Input image height')
    parser.add_argument('--vcm-test-batch-size', dest='vcm_test_batch_size', type=int, default=32)
    parser.add_argument('--seed', dest='seed', type=int, default=1)
    parser.add_argument('--num-train-ids', dest='num_train_ids', type=int, default=500, help='Number of training identities')
    parser.add_argument('--steps-per-epoch', dest='steps_per_epoch', type=int, default=200)
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, default=0.00004)
    parser.add_argument('--alignment-learning-rate', dest='alignment_learning_rate', type=float, default=0.0000004)
    parser.add_argument('--lr-multiplier', dest='lr_multiplier', type=int, default=25)
    parser.add_argument('--num-workers', dest='num_workers', default=8, type=int,
                        help='DataLoader worker count')

    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=1e-4)
    parser.add_argument('--output-dir', dest='output_dir', type=str, default='./logs/vcm/',
                        help='path to save related informations')
    parser.add_argument('--ciw-epochs', dest='ciw_epochs', type=int, default=30)
    parser.add_argument('--intra-epochs', dest='intra_epochs', type=int, default=30)
    parser.add_argument('--pgur-epochs', dest='pgur_epochs', type=int, default=50)
    parser.add_argument('--eval-interval', dest='eval_interval', type=int, default=2)
    parser.add_argument('--checkpoint-dir', dest='checkpoint_dir', type=str, default='./logs/vcm/models')
    parser.add_argument('--temporal-prompt-start-layer', dest='temporal_prompt_start_layer', type=int, default=9, help='Transformer block at which temporal prompts are inserted')

    parser.add_argument('--train-frame-sampling', dest='train_frame_sampling', type=str, default='random')
    parser.add_argument('--sequence-length', dest='sequence_length', type=int, default=6)
    parser.add_argument('--disable-random-flip', dest='random_flip', action='store_false', default=True)
    parser.add_argument('--test-frame-sampling', dest='test_frame_sampling', type=str, default='uniform')
    parser.add_argument('--test-sampler', dest='test_sampler', type=str,
                             default='ConsistentModalitySampler', help='None for no shuffle')
    parser.add_argument('--bupt-test-batch-size', dest='bupt_test_batch_size', type=int, default=64)  # Please don't change it.
    parser.add_argument('--train-sampler', dest='train_sampler', type=str,
                             default='RandomIdentitySampler', help='None for shuffle')
    parser.add_argument('--bupt-train-batch-size', dest='bupt_train_batch_size', type=int, default=16)
    parser.add_argument('--cameras-per-identity', dest='cameras_per_identity', type=int, default=2)
    parser.add_argument('--tracklets-per-camera', dest='tracklets_per_camera', type=int, default=1)
    parser.add_argument('--max-rank', dest='max_rank', type=int, default=20)


    # Full CBA configuration; all CIW and PGUR components remain enabled.
    parser.set_defaults(
        use_stp=True, alternate=False, fake=False,
        pgur_ir_weight=1.9, pgur_rgb_weight=1.5,
        ttb_weight=2.6, ics_weight=0.1, mpb_weight=1.0,
    )
    return parser




if __name__ == "__main__":
    config = build_parser().parse_args()
    seed_torch(config.seed)

    main(config)
