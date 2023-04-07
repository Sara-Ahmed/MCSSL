import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import torchvision

from datasets import load_dataset, datasets_utils

import utils
import vision_transformer as vits
import losses

def get_args_parser():
    parser = argparse.ArgumentParser('MCSSL', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='vit_small', type=str, choices=['vit_tiny', 'vit_small', 'vit_base'], help="architecture name.")
    parser.add_argument('--img_size', default=224, type=int, help="size of the input image.")
    parser.add_argument('--patch_size', default=16, type=int, help="size of the patch.")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    
    # Reconstruction head parameters
    parser.add_argument('--drop_perc', type=float, default=0.5, help='Drop X percentage of the input image')
    parser.add_argument('--drop_replace', type=float, default=0.3, help='Replace X percentage of the input image')
    
    parser.add_argument('--drop_align', type=int, default=1, help='Align drop with patches')
    parser.add_argument('--drop_type', type=str, default='zeros', help='Drop Type.')
    parser.add_argument('--drop_only', type=int, default=1, help='Align drop with patches')

    # Projection head parameters for both cls and data tokens
    parser.add_argument('--out_dim', default=8192, type=int, help="Dimensionality of the class head output.")

    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float, help="Initial value for the teacher temperature")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help="Number of warmup epochs for the teacher temperature.")
    parser.add_argument('--teacher_temp', default=0.07, type=float, help="Final value of the teacher temperature.") 
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="Base EMA parameter for teacher update.")

    # Training/Optimization parameters
    parser.add_argument('--weight_decay', type=float, default=0.04, help="Initial WD")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="Final WD")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="Maximal parameter gradient norm")
    
    parser.add_argument('--batch_size', default=8, type=int, help='batch size per GPU.')
    parser.add_argument('--epochs', default=800, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.0005, type=float, help="Learning rate.")
    
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="Fixing the output layer for N epochs")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="Target LR at the end of optimization.")
    
    # Dataset
    parser.add_argument('--data_set', default='Flowers', type=str, 
                        choices=['MNIST', 'CIFAR10', 'CIFAR100', 'Flowers', 'Aircraft', 'Cars', 'ImageNet5p', 'ImageNet10p', 'ImageNet', 'TinyImageNet', 'PASCALVOC', 'MSCOCO', 'VGenome', 'Pets', 'CUB'], 
                        help='Name of the dataset.')
    parser.add_argument('--data_location', default='/vol/research/facer2vm_fmad/people/sara/Transformer/datasets/', type=str, 
                        help='Location of the training dataset') 
    
    # Multi-crop parameters
    parser.add_argument('--global_crops_number', type=int, default=2, help="Number of global views.")
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.25, 1.),
        help="Scale range of the cropped image before resizing")
    parser.add_argument('--local_crops_number', type=int, default=10, help="Number of small local views.")
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.25),
        help="Scale range of the cropped image before resizing")
        
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser



class collate_batch(object): # replace from other images
    def __init__(self, drop_replace=0., drop_align=1):
        self.drop_replace = drop_replace
        self.drop_align = drop_align
        
    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        
        if self.drop_replace > 0:
            batch[0][1][0], batch[0][2][0] = datasets_utils.GMML_replace_list(batch[0][0][0], batch[0][1][0], batch[0][2][0],
                                                                            max_replace=self.drop_replace, align=self.drop_align)
            batch[0][1][1], batch[0][2][1] = datasets_utils.GMML_replace_list(batch[0][0][1], batch[0][1][1], batch[0][2][1],
                                                                            max_replace=self.drop_replace, align=self.drop_align)
        
        return batch

def train_MCSSL(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    ### Preparing data 
    transform = datasets_utils.DataAugmentation(args)   
    dataset = torchvision.datasets.ImageFolder(args.data_location, transform=transform)

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader( dataset, sampler=sampler, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_batch(args.drop_replace, args.drop_align))
    print(f"Data loaded: there are {len(dataset)} images.")

    ### building networks
    student = vits.__dict__[args.model](patch_size=args.patch_size, img_size=[args.img_size], drop_path_rate=args.drop_path_rate)
    teacher = vits.__dict__[args.model](patch_size=args.patch_size, img_size=[args.img_size])
    embed_dim = student.embed_dim

    # Build training pipeline
    student = FullModelPipline(student, vits.PROJHead(embed_dim, args.out_dim),
                                        vits.RECHead(embed_dim, patch_size=args.patch_size))
    teacher = FullModelPipline(teacher, vits.PROJHead(embed_dim, args.out_dim),
                                        vits.RECHead(embed_dim, patch_size=args.patch_size))

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    
    teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    ### Loss & optimizer
    calc_loss = losses.CALCULATELoss(args.out_dim, args.warmup_teacher_temp,
        args.teacher_temp, args.warmup_teacher_temp_epochs, args.epochs).cuda()
    
    recons_loss = nn.MSELoss(reduction='none').cuda()

    params_groups = utils.get_params_groups(student)
    optimizer = torch.optim.AdamW(params_groups)  

    fp16_scaler = torch.cuda.amp.GradScaler()

    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr, args.epochs, len(data_loader), warmup_epochs=args.warmup_epochs)
    wd_schedule = utils.cosine_scheduler(args.weight_decay,
        args.weight_decay_end, args.epochs, len(data_loader))

    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))

    # Resume training if checkpoint.pth exist
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student, teacher=teacher,
        optimizer=optimizer, fp16_scaler=fp16_scaler,
        calc_loss=calc_loss)
    start_epoch = to_restore["epoch"]
    
    start_time = time.time()
    print("Starting training !")
    for epoch in range(start_epoch, args.epochs+1):
        data_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, calc_loss, recons_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch, fp16_scaler, args)

        save_dict = {
            'student': student.state_dict(), 'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1, 'args': args,
            'calc_loss': calc_loss.state_dict()}
        
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
            
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, calc_loss, recons_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch, fp16_scaler, args):
    
    save_recon = os.path.join(args.output_dir, 'reconstruction_samples')
    Path(save_recon).mkdir(parents=True, exist_ok=True)
    plot_ = True
    
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, ((orig_imgs, corr_imgs, masks), _) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        corr_imgs = [im.cuda(non_blocking=True) for im in corr_imgs]
        masks = [im.cuda(non_blocking=True) for im in masks]
        orig_imgs = [im.cuda(non_blocking=True) for im in orig_imgs]
        
        
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            
            # global crops
            t_cls, t_data, _        = teacher(orig_imgs[:2]) 
            s_cls, s_data, s_recons = student(corr_imgs[:2], recons=True)
            
            # local crops
            l_cls, _, _ = student(orig_imgs[2:])
            
            # Classification loss ------------------------------------------------
            c_loss, p_loss = calc_loss(torch.cat((s_cls, l_cls), dim=0), t_cls, s_data, t_data, epoch)
            
            # Recons Loss -------------------------------------------------
            rloss = recons_loss(s_recons, torch.cat(orig_imgs[:2]))
            r_loss = rloss[torch.cat(masks[0:])==1].mean() 
                
            if plot_==True and utils.is_main_process():# and args.saveckp_freq and epoch % args.saveckp_freq == 0:
                plot_ = False
                #validating: check the reconstructed images
                print_out = save_recon + '/epoch_' + str(epoch).zfill(5)  + '.jpg' 
                imagesToPrint = torch.cat([orig_imgs[0][0: min(15, args.batch_size)].cpu(),  corr_imgs[0][0: min(15, args.batch_size)].cpu(),
                                       s_recons[0: min(15, args.batch_size)].cpu(), masks[0][0: min(15, args.batch_size)].cpu()], dim=0)
                torchvision.utils.save_image(imagesToPrint, print_out, nrow=min(15, args.batch_size), normalize=True, range=(-1, 1))
                        
            loss = c_loss + p_loss + r_loss 

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(r_loss=r_loss.item())
        metric_logger.update(c_loss=c_loss.item())
        metric_logger.update(p_loss=p_loss.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



class FullModelPipline(nn.Module):

    def __init__(self, backbone, head, head_recons):
        super(FullModelPipline, self).__init__()
        
        self.backbone = backbone
        self.head = head
        self.head_recons = head_recons
                
    def forward(self, x, recons=False):
        _out = self.backbone(torch.cat(x[0:]))
        
        recons_imgs = self.head_recons(_out[:, 1:]) if recons == True else None
        
        cls_ftrs, data_ftrs = self.head(_out)
        
        return  cls_ftrs, data_ftrs, recons_imgs



if __name__ == '__main__':
    parser = argparse.ArgumentParser('MCSSL', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_MCSSL(args)
