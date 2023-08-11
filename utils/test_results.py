import os
import time

import pandas as pd
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from metrics.metricstools import metrics


def test(model, dataloader, local_rank, instance_name, test_log_save_dir,
               metrics_measures=None, mertric_mode=None, mertric_reduction=None, seg_threshold=None,
               mertric_ignore_index=None, mertric_num_classes=None):
    iteration = 0

    tensorboard_log_save_dir = os.path.join(test_log_save_dir, instance_name)
    writer = SummaryWriter(log_dir=tensorboard_log_save_dir)

    num_test_imgs = len(dataloader.dataset)

    model.to(local_rank)

    t_epoch_start = time.time()
    t_iter_start = time.time()

    iteration_test_prec = 0.0
    iteration_test_rec = 0.0
    iteration_test_acc = 0.0
    iteration_test_F1 = 0.0
    iteration_test_IoU = 0.0

    model.eval()
    print('--------------')
    print('(test)')

    for imgs, targets in tqdm(dataloader, ncols=150, colour='#C0FF20'):
        if imgs.size()[0] == 1:
            continue

        imgs = imgs.to(local_rank)
        targets = targets.to(local_rank)

        outputs = model(imgs)

        t_iter_finish = time.time()
        iter_Time = t_iter_finish - t_iter_start

        t_iter_start = time.time()

        probs = outputs[0].to(torch.float32) if model.use_deep_supervision else outputs.to(torch.float32)

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

        if iteration % 5 == 0:
            print(
                "=================================================================================================" * 2)

            print(
                ' ————————— iteration - {} || IoU: {:.4f} || F1: {:.4f} || Pre: {:.4f} || Rec: {:.4f} || Acc: {:.4f}.'.format(
                    iteration,
                    metrics_results['IoU'].item(),
                    metrics_results['F1'].item(),
                    metrics_results['Pre'].item(),
                    metrics_results['Rec'].item(),
                    metrics_results['Acc'].item(),
                ))
            t_iter_start = time.time()

            print(
                "============================================================================================================================" * 2)


        # writer.add_scalar('loss', loss.item(), iteration)
        writer.add_scalar('F1', metrics_results['F1'].item(), iteration)
        writer.add_scalar('IoU', metrics_results['IoU'].item(), iteration)
        writer.add_scalar('Pre', metrics_results['Pre'].item(), iteration)
        writer.add_scalar('Rec', metrics_results['Rec'].item(), iteration)
        writer.add_scalar('Acc', metrics_results['Acc'].item(), iteration)
        writer.add_scalar('iter_Time', iter_Time, iteration)

        iteration_test_F1 += metrics_results['F1'].item() * imgs.size()[0]
        iteration_test_IoU += metrics_results['IoU'].item() * imgs.size()[0]
        iteration_test_prec += metrics_results['Pre'].item() * imgs.size()[0]
        iteration_test_rec += metrics_results['Rec'].item() * imgs.size()[0]
        iteration_test_acc += metrics_results['Acc'].item() * imgs.size()[0]

        iteration += 1

    t_epoch_finish = time.time()

    micro_epoch_test_F1 = iteration_test_F1 / num_test_imgs
    micro_epoch_test_IoU = iteration_test_IoU / num_test_imgs
    micro_epoch_test_prec = iteration_test_prec / num_test_imgs
    micro_epoch_test_rec = iteration_test_rec / num_test_imgs
    micro_epoch_test_acc = iteration_test_acc / num_test_imgs

    epoch_Time = t_epoch_finish - t_epoch_start

    ################################################################################


    print(4 * "-------")

    print('|| Epoch_test_F1: {:.4f}'.format(
        micro_epoch_test_F1
    ))

    print(4 * "-------")

    print('|| Epoch_test_IoU: {:.4f}'.format(
        micro_epoch_test_IoU
    ))

    print(4 * "-------")

    print('|| Epoch_test_prec: {:.4f}'.format(
        micro_epoch_test_prec
    ))

    print(4 * "-------")

    print('|| Epoch_test_rec: {:.4f}'.format(
        micro_epoch_test_rec
    ))

    print(4 * "-------")

    print('|| Epoch_test_acc: {:.4f}'.format(
        micro_epoch_test_acc
    ))

    print(4 * "-------")

    print('|| epoch_Timer: {:.4f} sec.'.format(epoch_Time))

    print(4 * "-------")

    log = {
        'model_name': instance_name,
        'Epoch_test_F1': micro_epoch_test_F1,
        'Epoch_test_IoU': micro_epoch_test_IoU,
        'Epoch_test_prec': micro_epoch_test_prec,
        'Epoch_test_rec': micro_epoch_test_rec,
        'Epoch_test_acc': micro_epoch_test_acc,
        'epoch_Timer': epoch_Time,
    }
    df = pd.DataFrame(log, index=[0])
    log_path = tensorboard_log_save_dir + ".csv"
    df.to_csv(log_path, float_format='%.4f')
    writer.close()
