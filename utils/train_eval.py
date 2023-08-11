import logging
from logging.handlers import TimedRotatingFileHandler
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
import os
import time
import math
import torch
import glo
import torch.nn as nn
from tqdm import tqdm
from typing import List, Dict
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from metrics.metricstools import metrics
from utils.dataset_and_loader import get_dataloader
from utils.train_resume_stop import resume_training_fn, TrainingStatus

def train_model(model: nn.Module,
                dataset_dict: Dict,
                loss_fn,
                optimizer,
                scheduler,
                work_id,
                # instance_name: str,
                use_deep_supervision: bool,
                deep_supervision_weights: List[float],
                num_epochs: int,
                batch_size: int,
                val_interval: int,
                local_rank,
                num_workers: int,
                seg_threshold: float,
                use_distributed: bool,
                use_amp: bool,
                metrics_measures: List[str],
                mertric_mode: str,
                mertric_reduction: str,
                mertric_ignore_index: int,
                mertric_num_classes: int,
                predicted_checkpoint_path: str,
                checkpoint_save_dir: str,
                tensorboard_save_dir: str,
                # logging_save_dir: str,
                logger: logging.Logger,
                terminal_log_print_interval: int
                ) -> None:
    """
    Startup program for model training, supporting distributed training, 
    automatic precision mixed training, tensorboard log printing and other functions

    Args:
        mertric_num_classes:
        mertric_ignore_index:
        mertric_reduction:
        mertric_mode:
        model (nn.Module): segmentation model
        dataset_dict (Dict): path to Dataset
        loss_fn (_type_): loss function type
        optimizer (_type_): optimizer
        scheduler (_type_): learning rate scheduler policy
        instance_name (str): the name of the running instance
        use_deep_supervision (bool): whether to use deep supervision
        deep_supervision_weights (List[float]): weights of deep_supervision
        num_epochs (int): number of epochs
        batch_size (int): batch size
        val_interval (int): interval of val 
        local_rank (_type_): rank of device
        num_workers (int): Specifies the number of worker processes for the data loader.
        seg_threshold (float): segmentation threshold defaults to single layer output
        use_distributed (bool): whether to perform distributed training
        use_amp (bool): hether to use amp(automatic mixing accuracy training)
        metrics_measures (List[str]): result accuracy evaluation index
        predicted_checkpoint_path (str): path of complete weights for the model
        checkpoint_save_dir (str): path to save model checkpoint dir
        tensorboard_save_dir (str): save path of tensorboard log file
        logging_save_dir (str): save path of logging file
        terminal_log_print_interval (int): print interval of terminal log

    Raises:
        FileNotFoundError: resume model checkpoint path do not exist!
        FileNotFoundError: Pretrained weights do not exist!
    """


    logger.info("---------------------------------------- Training started ----------------------------------------")
    logger.info("\n")
    logger.info(f"=================================正在运行的是workid：{work_id} ============================================")
    logger.info("\n")

    try:
        from pprint import pprint
        logger.info(f"===========------------Name of model:{model.name}------------===========")
    except ModuleNotFoundError:
        logger.error("model is not exist!")

    if not os.path.exists(tensorboard_save_dir):
        os.makedirs(tensorboard_save_dir)
    writer = SummaryWriter(log_dir=tensorboard_save_dir)

    start_epoch = 1

    best_micro_epoch_train_F1 = 0.0
    best_micro_epoch_train_IoU = 0.0
    best_micro_epoch_val_F1 = 0.0
    best_micro_epoch_val_IoU = 0.0

    checkpoint_save_path = checkpoint_save_dir


    if not os.path.exists(checkpoint_save_path):
        os.makedirs(checkpoint_save_path)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if glo.get_value([work_id, 'train_state']) == TrainingStatus.STOPPED and glo.get_value([work_id, 'resume_flag']) == True:
        resume_model_checkpoint_path = os.path.join(checkpoint_save_path, 'last_epoch_to_RESUM_train.pth')
        if os.path.exists(resume_model_checkpoint_path):
            checkpoint = torch.load(resume_model_checkpoint_path, map_location=torch.device('cpu'))

            resume_training_fn(checkpoint, model, optimizer, scheduler, scaler, local_rank, use_amp)
            last_epoch = checkpoint['epoch']
            start_epoch = last_epoch + 1

            logger.info(
                '################################ -------- training resumed! -------- ################################')
            logger.info(
                '--------------------------------- last_epoch: {} || start_epoch: {}! ---------------------------------'.format(
                    last_epoch, start_epoch))

            predicted_checkpoint_path = None

        else:
            logger.warning(
                f"resume model checkpoint path [{resume_model_checkpoint_path}] do not exist! Defalut to retraining!")

        glo.set_value([work_id, 'resume_flag'], False)
        glo.set_value([work_id, 'train_state'], TrainingStatus.TRINGING)

    if predicted_checkpoint_path is not None:
        if os.path.exists(predicted_checkpoint_path):
            checkpoint = torch.load(predicted_checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint["model"])
            logger.info(f"predicted checkpoint loaded!")
            model.to(local_rank)
        else:
            logger.error(f"Pretrained weights [{predicted_checkpoint_path}] do not exist!")
            raise FileNotFoundError(f"Pretrained weights [{predicted_checkpoint_path}] do not exist!")
    else:
        model.to(local_rank)

    if use_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    num_train_imgs = dataset_dict["train"].__len__()
    num_val_imgs = dataset_dict["val"].__len__()
    initial_iteration = (start_epoch - 1) * math.ceil(num_train_imgs / batch_size) + 1
    iteration = initial_iteration

    dataloader_dict, train_sampler = get_dataloader(dataset_dict=dataset_dict,
                                                    batch_size=batch_size,
                                                    is_distributed=use_distributed,
                                                    num_workers=num_workers)

    for epoch in range(start_epoch, num_epochs + 1):

        if use_distributed:
            train_sampler.set_epoch(epoch)

        t_epoch_start = time.time()
        t_iter_start = time.time()

        epoch_train_loss = 0.0
        epoch_train_F1 = 0.0
        epoch_train_IoU = 0.0
        epoch_train_prec = 0.0
        epoch_train_rec = 0.0
        epoch_train_acc = 0.0

        epoch_val_loss = 0.0
        epoch_val_F1 = 0.0
        epoch_val_IoU = 0.0
        epoch_val_prec = 0.0
        epoch_val_rec = 0.0
        epoch_val_acc = 0.0

        print('\n')

        logger.info(
            '*************************************     Epoch{} / {}     *************************************'.format(
                epoch, num_epochs))

        print('\n')
        for phase in ["train", "val"]:

            if phase == "train":
                model.train()
                logger.info(
                    "=========================================" * 2)                  
                logger.info(
                    '------------------------------------(train--phase)-------------------------------------')
                logger.info(
                    "=========================================" * 2)  
                print('\n')
            else:
                if epoch % val_interval == 0:
                    model.eval()
                    print('\n')
                    logger.info(
                        "=========================================" * 2)                  
                    logger.info(
                        '------------------------------------(val--phase)-------------------------------------')
                    logger.info(
                        "=========================================" * 2)  
                    print('\n')
                else:
                    continue

            logger.info(
                "====================================================================================")
            
            for imgs, targets in tqdm(dataloader_dict[phase], ncols=150, colour='#C0FF20'):
                print("\n")
                if (glo.get_value([work_id, 'train_state']) == TrainingStatus.TRINGING and glo.get_value([work_id, 'stop_flag']) == True):
                    print('\n')
                    logger.info(
                        "################################# -------- training stoped! -------- #################################")
                    print('\n')
                    glo.set_value([work_id, 'stop_flag'], False)
                    glo.set_value([work_id, 'train_state'], TrainingStatus.STOPPED)

                    return 

                if imgs.size()[0] == 1:
                    continue

                imgs = imgs.to(local_rank)
                targets = targets.to(local_rank)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                        outputs = model(imgs)
                        loss_fn = loss_fn.to(local_rank)

                        if use_deep_supervision:

                            assert len(deep_supervision_weights) == len(outputs)

                            assert all(isinstance(element, float or int) for element in deep_supervision_weights)

                            loss = sum(list(map(lambda x, y, z: z * loss_fn(x, y), outputs,
                                                [targets.unsqueeze(1).float() if model.num_classes == 1 else targets] * len(outputs),
                                                deep_supervision_weights)))
                        else:
                            loss = loss_fn(outputs, targets.unsqueeze(1).float() if model.num_classes == 1 else targets)

                    probs = outputs[0].to(torch.float32) if use_deep_supervision else outputs.to(torch.float32)

                    if model.num_classes == 1:
                        out = torch.sigmoid(probs)
                    else:
                        out = torch.argmax(torch.softmax(probs, dim=1), dim=1)

                    metrics_results = metrics(out,
                                              targets.unsqueeze(1) if model.num_classes == 1 else targets,
                                              metrics_measures=metrics_measures,
                                              mode=mertric_mode,
                                              threshold=seg_threshold,
                                              reduction=mertric_reduction,
                                              ignore_index=mertric_ignore_index,
                                              num_classes=mertric_num_classes
                                              )

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()


                        if iteration % terminal_log_print_interval == 0:
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start

                            print('\n')

                            logger.info(
                                ' >>> epoch - {} || iteration - {} || Loss: {:.4f} || 10iter: {:.4f} sec || lr: {:.8f} || IoU: {:.4f} || F1: {:.4f} || Pre: {:.4f} || Rec: {:.4f} || Acc: {:.4f}. <<<'.format(
                                    epoch,
                                    iteration,
                                    loss.item() / batch_size * imgs.size()[0],
                                    duration,
                                    optimizer.param_groups[0]['lr'],
                                    metrics_results['IoU'].item(),
                                    metrics_results['F1'].item(),
                                    metrics_results['Pre'].item(),
                                    metrics_results['Rec'].item(),
                                    metrics_results['Acc'].item(),
                                ))
                            t_iter_start = time.time()

                            logger.info(
                                "====================================================================================")

                        epoch_train_loss += loss.item() * imgs.size()[0]
                        epoch_train_F1 += metrics_results['F1'].item() * imgs.size()[0]
                        epoch_train_IoU += metrics_results['IoU'].item() * imgs.size()[0]
                        epoch_train_prec += metrics_results['Pre'].item() * imgs.size()[0]
                        epoch_train_rec += metrics_results['Rec'].item() * imgs.size()[0]
                        epoch_train_acc += metrics_results['Acc'].item() * imgs.size()[0]

                        iteration += 1
                    else:
                        epoch_val_loss += loss.item() * imgs.size()[0]
                        epoch_val_F1 += metrics_results['F1'].item() * imgs.size()[0]
                        epoch_val_IoU += metrics_results['IoU'].item() * imgs.size()[0]
                        epoch_val_prec += metrics_results['Pre'].item() * imgs.size()[0]
                        epoch_val_rec += metrics_results['Rec'].item() * imgs.size()[0]
                        epoch_val_acc += metrics_results['Acc'].item() * imgs.size()[0]

            if phase == 'train':
                scheduler.step(epoch)

            t_epoch_finish = time.time()

            micro_epoch_train_loss = epoch_train_loss / num_train_imgs
            micro_epoch_val_loss = epoch_val_loss / num_val_imgs
            micro_epoch_train_prec = epoch_train_prec / num_train_imgs
            micro_epoch_val_prec = epoch_val_prec / num_val_imgs
            micro_epoch_train_rec = epoch_train_rec / num_train_imgs
            micro_epoch_va_rec = epoch_val_rec / num_val_imgs
            micro_epoch_train_acc = epoch_train_acc / num_train_imgs
            micro_epoch_val_acc = epoch_val_acc / num_val_imgs
            micro_epoch_train_F1 = epoch_train_F1 / num_train_imgs
            micro_epoch_val_F1 = epoch_val_F1 / num_val_imgs
            micro_epoch_train_IoU = epoch_train_IoU / num_train_imgs
            micro_epoch_val_IoU = epoch_val_IoU / num_val_imgs

            Time = t_epoch_finish - t_epoch_start

            if phase == 'val':
                logger.info(
                    "====================================================================================")

                ################################################################################
                logger.info('epoch {} || Epoch_TRAIN_Loss: {:.4f} || Epoch_VAL_LOSS: {:.4f}'.format(
                    epoch, micro_epoch_train_loss, micro_epoch_val_loss
                ))

                logger.info('epoch {} || Epoch_TRAIN_F1: {:.4f} || Epoch_VAL_F1: {:.4f}'.format(
                    epoch, micro_epoch_train_F1, micro_epoch_val_F1
                ))

                logger.info('epoch {} || Epoch_TRAIN_IoU: {:.4f} || Epoch_VAL_IoU: {:.4f}'.format(
                    epoch, micro_epoch_train_IoU, micro_epoch_val_IoU
                ))

                logger.info('epoch {} || Epoch_TRAIN_PREC: {:.4f} || Epoch_VAL_PREC: {:.4f}'.format(
                    epoch, micro_epoch_train_prec, micro_epoch_val_prec
                ))

                logger.info('epoch {} || Epoch_TRAIN_REC: {:.4f} || Epoch_VAL_REC: {:.4f}'.format(
                    epoch, micro_epoch_train_rec, micro_epoch_va_rec
                ))

                logger.info('epoch {} || Epoch_TRAIN_ACC: {:.4f} || Epoch_VAL_ACC: {:.4f}'.format(
                    epoch, micro_epoch_train_acc, micro_epoch_val_acc
                ))

                logger.info('Timer: {:.4f} sec.'.format(Time))

                writer.add_scalars('loss', {'train_loss': micro_epoch_train_loss,
                                            'val_loss': micro_epoch_val_loss}, epoch)

                writer.add_scalars('prec', {'train_prec': micro_epoch_train_prec,
                                            'val_prec': micro_epoch_val_prec}, epoch)

                writer.add_scalars('rec',
                                   {'train_rec': micro_epoch_train_rec,
                                    'val_rec': micro_epoch_va_rec}, epoch)

                writer.add_scalars('acc',
                                   {'train_acc': micro_epoch_train_acc,
                                    'val_acc': micro_epoch_val_acc}, epoch)

                writer.add_scalars('F1', {'train_F1': micro_epoch_train_F1,
                                          'val_F1': micro_epoch_val_F1}, epoch)

                writer.add_scalars('IoU', {'train_IoU': micro_epoch_train_IoU,
                                           'val_IoU': micro_epoch_val_IoU}, epoch)

                writer.add_scalar('Time', Time, epoch)

            checkpoints = {
                'epoch': epoch,
                'model': model.state_dict() if not use_distributed else model.module.static_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict()
            }

            if not use_amp:
                checkpoints.pop("scaler")
            
            if phase == 'train':
                if epoch != 1:
                    os.remove(os.path.join(checkpoint_save_path, 'last_epoch_to_RESUM_train.pth'))
                torch.save(checkpoints,
                           os.path.join(checkpoint_save_path, 'last_epoch_to_RESUM_train.pth'))
                
                checkpoints.pop('optimizer')
                checkpoints.pop('scheduler')
                if use_amp:
                    checkpoints.pop("scaler")
                if micro_epoch_train_F1 > best_micro_epoch_train_F1:
                    best_micro_epoch_train_F1 = micro_epoch_train_F1
                    torch.save(checkpoints,
                               os.path.join(checkpoint_save_path, str('train_best_f1.pth')))

                if micro_epoch_train_IoU > best_micro_epoch_train_IoU:
                    best_micro_epoch_train_IoU = micro_epoch_train_IoU
                    torch.save(checkpoints,
                               os.path.join(checkpoint_save_path, str('train_best_iou.pth')))

            if phase == 'val':

                checkpoints.pop('optimizer')
                checkpoints.pop('scheduler')
                if use_amp:
                    checkpoints.pop("scaler") 

                if micro_epoch_val_F1 > best_micro_epoch_val_F1:
                    best_micro_epoch_val_F1 = micro_epoch_val_F1
                    torch.save(checkpoints,
                               os.path.join(checkpoint_save_path, str('val_best_f1.pth')))
                if micro_epoch_val_IoU > best_micro_epoch_val_IoU:
                    best_micro_epoch_val_IoU = micro_epoch_val_IoU
                    torch.save(checkpoints,
                               os.path.join(checkpoint_save_path, str('val_best_iou.pth')))
                    
    writer.close()
    glo.set_value([work_id, 'train_state'], TrainingStatus.FINISHED)
    logger.info("--------------------------------- Training finished --------------------------------")
