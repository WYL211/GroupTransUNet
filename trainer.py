import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from networks.model import Model
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume


def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item

# def trainer_synapse(args, model, snapshot_path):
#     from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
#     logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
#                         format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
#     logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
#     logging.info(str(args))
#     base_lr = args.base_lr
#     num_classes = args.num_classes
#     batch_size = args.batch_size * args.n_gpu
#     # max_iterations = args.max_iterations
#     db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
#                                transform=transforms.Compose(
#                                    [RandomGenerator(output_size=[args.img_size, args.img_size])]))
#     print("The length of train set is: {}".format(len(db_train)))
#
#     def worker_init_fn(worker_id):
#         random.seed(args.seed + worker_id)
#
#     trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
#                              worker_init_fn=worker_init_fn)
#     if args.n_gpu > 1:
#         model = nn.DataParallel(model)
#     model.train()
#     ce_loss = CrossEntropyLoss()
#     dice_loss = DiceLoss(num_classes)
#     optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
#
#     writer = SummaryWriter(snapshot_path + '/log')
#     iter_num = 0
#     max_epoch = args.max_epochs
#     max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
#     # warm_up_iterations = args.warmup * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
#     logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
#     best_performance = 0.0
#     iterator = tqdm(range(max_epoch), ncols=70)
#     # lr_ = base_lr
#
#     l = [0, 1, 2, 3]
#
#     ss = [x for x in powerset(l)]
#
#     # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)
#     scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
#     for epoch_num in iterator:
#
#
#         for i_batch, sampled_batch in enumerate(trainloader):
#             image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
#             image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
#             # print("ttttttttttttt",image_batch.shape)
#             outputs = model(image_batch)
#
#
#             loss = 0.0
#             lc1, lc2 = 0.3, 0.7
#
#
#             for s in ss:
#
#                 iout = 0.0
#                 if (s == []):
#                     continue
#                 for idx in range(len(s)):
#
#                     iout += outputs[s[idx]]
#
#                 loss_ce = ce_loss(iout, label_batch)
#                 loss_dice = dice_loss(iout, label_batch, softmax=True)
#                 loss += (lc1 * loss_ce + lc2 * loss_dice)
#
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             writer.add_scalar('info/total_loss', loss, iter_num)
#             writer.add_scalar('info/loss_ce', loss_ce, iter_num)
#
#
#
#
#             img = 0.0
#             for idx in range(len(outputs)):
#                 img += outputs[idx]
#             if iter_num % 20 == 0:
#                 image = image_batch[1, 0:1, :, :]
#                 image = (image - image.min()) / (image.max() - image.min())
#                 writer.add_image('train/Image', image, iter_num)
#                 outputs = torch.argmax(torch.softmax(img, dim=1), dim=1, keepdim=True)
#                 writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
#                 labs = label_batch[1, ...].unsqueeze(0) * 50
#                 writer.add_image('train/GroundTruth', labs, iter_num)
#
#             iter_num+=1
#
#
#
#         scheduler.step()
#         print('lr', optimizer.state_dict()['param_groups'][0]['lr'])
#
#
#
#
#
#         save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
#         torch.save(model.state_dict(), save_mode_path)
#         logging.info("save model to {}".format(save_mode_path))
#
#
#
#         if epoch_num >= max_epoch - 1:
#             save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
#             torch.save(model.state_dict(), save_mode_path)
#             logging.info("save model to {}".format(save_mode_path))
#             iterator.close()
#             break
#
#     writer.close()
#     return "Training Finished!"


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations

    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", img_size=224,  #这个图片尺寸是新添加的
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    # db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir,img_size=224)  # 这个图片尺寸是新添加的
    # testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    # print("The test iterations per epoch is: {}".format(len(testloader)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=5E-3)

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    # warm_up_iterations = args.warmup * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    # lr_ = base_lr

    l = [0, 1, 2, 3]
    ss = [x for x in powerset(l)]
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)


    for epoch_num in iterator:
        total_loss =0
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            # image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()
            outputs = model(image_batch)
            loss = 0.0
            lc1, lc2 = 0.4, 0.6
            for s in ss:
                iout = 0.0
                if (s == []):
                    continue
                for idx in range(len(s)):
                    iout += outputs[s[idx]]
                loss_ce = ce_loss(iout, label_batch[:].long())   #
                loss_dice = dice_loss(iout, label_batch, softmax=True)
                loss += (lc1 * loss_ce + lc2 * loss_dice)





            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            # logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            img = 0.0
            for idx in range(len(outputs)):
                img += outputs[idx]
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(img, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
            iter_num+=1
        scheduler.step()
        print('lr', optimizer.state_dict()['param_groups'][0]['lr'])


        # ------------------Validation--------------
        # if epoch_num >= int(max_epoch / 2):
        #     model.eval()
        #     with torch.no_grad():
        #         metric_list = 0.0
        #         for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        #             h, w = sampled_batch["image"].size()[2:]
        #             image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        #             metric_i = test_single_volume(image, label, model, classes=args.num_classes,
        #                                           patch_size=[args.img_size, args.img_size],
        #                                           test_save_path=None, case=case_name, z_spacing=args.z_spacing)
        #             metric_list += np.array(metric_i)
        #             # print('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        #             # logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        #             # file.write('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]) + '\n')
        #         metric_list = metric_list / len(db_test)
        #         for i in range(1, args.num_classes):
        #             logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
        #             print('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
        #             # file.write('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]) + '\n')
        #
        #         performance = np.mean(metric_list, axis=0)[0]
        #         mean_hd95 = np.mean(metric_list, axis=0)[1]
        #         logging.info('valid performance: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
        #         print('valid performance: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
        #         if performance > best_performance:
        #             best_performance = performance
        #             save_mode_path = os.path.join(snapshot_path, 'best_epoch_' + str(epoch_num) + '.pth')
        #             torch.save(model.state_dict(), save_mode_path)
        # if epoch_num >=max_epoch -1:
        #     print(best_performance)


        if epoch_num >= int(max_epoch / 2):
        # if epoch_num >= 400:

            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break


    writer.close()
    return "Training Finished!"\





