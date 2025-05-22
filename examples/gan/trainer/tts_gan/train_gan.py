from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import cfg
import os
import datetime
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data.distributed
from torch.utils import data
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
from adamw import AdamW
import random 
import matplotlib.pyplot as plt
import io
import PIL.Image

from auto_retraining.model_repository.pytorch.pytorch_model_repository import PytorchModelRepository
from utils import set_log_dir, save_checkpoint, create_logger
from convergence import convergence
from gan_models import * 
from functions import train, save_samples, LinearLrDecay, load_params, copy_params, cur_stages
from torchvision.transforms import ToTensor
from trainer import DataSet


def main():
    args = cfg.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    main_worker(args.gpu, args)
        
def main_worker(gpu, args):
    args.gpu = gpu
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    seq_len = args.seq_len
    g_embed_dim = args.g_embed_dim
    d_embed_dim = args.d_embed_dim
    g_num_heads = args.g_num_heads
    d_num_heads = args.d_num_heads
    d_depth = args.d_depth
    g_depth = args.g_depth
    g_patch_size = args.g_patch_size
    d_patch_size = args.d_patch_size

    # import network
    gen_net = Generator(seq_len=args.seq_len, embed_dim=g_embed_dim, patch_size=g_patch_size, num_heads=g_num_heads, depth=g_depth)
    print(gen_net)
    dis_net = Discriminator(seq_length=args.seq_len, patch_size=d_patch_size, emb_size=d_embed_dim, depth=d_depth, num_heads=d_num_heads)
    print(dis_net)
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        gen_net.cuda()
        dis_net.cuda()
    print(dis_net) if args.rank == 0 else 0
    # set optimizer
    if args.optimizer == "adam":
        gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                        args.g_lr, (args.beta1, args.beta2))
        dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                        args.d_lr, (args.beta1, args.beta2))
    elif args.optimizer == "adamw":
        gen_optimizer = AdamW(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                        args.g_lr, weight_decay=args.wd)
        dis_optimizer = AdamW(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                         args.g_lr, weight_decay=args.wd)
        
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, args.max_iter * args.n_critic)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, args.max_iter * args.n_critic)

    # epoch number for dis_net
    args.max_epoch = args.max_epoch * args.n_critic
    
    train_set = DataSet(torch.load(args.training_set_path))
    train_loader = data.DataLoader(train_set,  batch_size=args.batch_size, num_workers=args.num_workers, shuffle = True)
    
    print(len(train_loader))
    
    if not args.max_epoch and args.max_iter:
        args.max_epoch = np.ceil(args.max_iter * args.n_critic / len(train_loader))

    # initial
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (100, args.latent_dim)))
    avg_gen_net = deepcopy(gen_net).cpu()
    gen_avg_param = copy_params(avg_gen_net)
    del avg_gen_net
    start_epoch = 0
    best_fid = 1e4

    # set writer
    writer = None
    if args.load_path and os.path.exists(args.load_path):
        print(f'=> resuming from {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint_file = os.path.join(args.load_path)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(0)
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        start_epoch = checkpoint['epoch']
        best_fid = checkpoint['best_fid']
        
        
        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        
        gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        gen_avg_param = copy_params(gen_net, mode='gpu')
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        fixed_z = checkpoint['fixed_z']
        
        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path']) if args.rank == 0 else None
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
        writer = SummaryWriter(args.path_helper['log_path']) if args.rank == 0 else None
        del checkpoint
    else:
    # create new log dir
        assert args.exp_name
        if args.rank == 0:
            args.path_helper = set_log_dir(args.logs_dir, args.exp_name)
            logger = create_logger(args.path_helper['log_path'])
            writer = SummaryWriter(args.path_helper['log_path'])
    
    if args.rank == 0:
        logger.info(args)
    writer_dict = {
        'writer': writer,
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }
    config_table = f"""
    | field | value  |
    |----------|-----------|
    | g_embed_dim    | {g_embed_dim:.2f} |
    | d_embed_dim    | {d_embed_dim:.2f} |
    | d_patch_size    | {d_patch_size:.2f} |
    | g_patch_size    | {g_patch_size:.2f} |
    | d_num_heads    | {d_num_heads:.2f} |
    | g_num_heads    | {g_num_heads:.2f} |
    | d_depth    | {d_depth:.2f} |
    | g_depth    | {g_depth:.2f} |
    """
    config_table = '\n'.join(l.strip() for l in config_table.splitlines())
    writer.add_text("table", config_table, 0)

    # train loop
    for epoch in range(int(start_epoch), int(args.max_epoch)):
#         train_sampler.set_epoch(epoch)
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        cur_stage = cur_stages(epoch, args)
        print("cur_stage " + str(cur_stage)) if args.rank==0 else 0
        print(f"path: {args.path_helper['prefix']}") if args.rank==0 else 0
        train(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict, fixed_z, lr_schedulers)
        
        if args.rank == 0 and args.show:
            backup_param = copy_params(gen_net)
            load_params(gen_net, gen_avg_param, args, mode="cpu")
            save_samples(args, fixed_z, epoch, gen_net, writer_dict)
            load_params(gen_net, backup_param, args)

        gen_net.eval()
        plot_buf = gen_plot(gen_net, epoch, args.class_name)
        image = PIL.Image.open(plot_buf)
        image = ToTensor()(image).unsqueeze(0)
        writer.add_image('Image', image[0], epoch)
        
        avg_gen_net = deepcopy(gen_net)
        load_params(avg_gen_net, gen_avg_param, args)
        print("!!!!!!!!", args.path_helper['ckpt_path'])
        save_checkpoint({
            'epoch': epoch + 1,
            'gen_model': args.gen_model,
            'dis_model': args.dis_model,
            'gen_state_dict': gen_net.state_dict(),
            'dis_state_dict': dis_net.state_dict(),
            'avg_gen_state_dict': avg_gen_net.state_dict(),
            'gen_optimizer': gen_optimizer.state_dict(),
            'dis_optimizer': dis_optimizer.state_dict(),
            'best_fid': best_fid,
            'path_helper': args.path_helper,
            'fixed_z': fixed_z
        }, args.path_helper['ckpt_path'], filename="checkpoint")
        if epoch % 10 == 0:
            conv = convergence(train_set.data, 
                           args.seq_len, 
                           avg_gen_net)
            writer.add_scalar('conv', conv, writer_dict['train_global_steps'])
        del avg_gen_net
    avg_gen_net = deepcopy(gen_net)
    conv = convergence(train_set.data, 
                           args.seq_len, 
                           avg_gen_net)
    writer.add_scalar('conv', conv, writer_dict['train_global_steps'])
    
    model_repository = PytorchModelRepository(experiment_name=args.exp_name)
    current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_repository.write(dis_net.state_dict(), "discriminator", current_date)
    model_repository.write(gen_net.state_dict(), "generator", current_date)
        
def gen_plot(gen_net, epoch, class_name):
    """Create a pyplot plot and save to buffer."""
    synthetic_data = [] 

    for i in range(10):
        fake_noise = torch.FloatTensor(np.random.normal(0, 1, (1, 100))).cuda()
        fake_sigs = gen_net(fake_noise).to('cpu').detach().numpy()
        synthetic_data.append(fake_sigs)

    fig, axs = plt.subplots(2, 5, figsize=(20,5))
    fig.suptitle(f'Synthetic {class_name} at epoch {epoch}', fontsize=30)
    for i in range(2):
        for j in range(5):
            axs[i, j].plot(synthetic_data[i*5+j][0][0][0][:])
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    return buf

if __name__ == '__main__':
    main()
