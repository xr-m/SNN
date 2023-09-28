import os
import torch
import logging
import model
import dataloaders
from tools import *
import numpy as np
import torch.nn as nn
import torch.optim as optim


def tst(myNet, test_data_loader):
    myNet.eval()
    test_correct = 0
    with torch.no_grad():
        for t_sample, t_label in test_data_loader:
            t_sample, t_label = t_sample.cuda(), t_label.cuda()
            class_output = myNet(t_sample)
            class_output = torch.argmax(class_output, dim=1)
            test_correct += torch.eq(class_output, t_label).float().sum().item()
    test_acc = test_correct / len(test_data_loader.dataset) * 100
    return test_acc


def train(myNet, source_loader):
    global loss_e
    myNet.train()
    correct = 0
    for source_data, source_label in source_loader:
        data_s, label_s_e = source_data.cuda(), source_label.cuda()
        label_s_e = label_s_e.long()
        s_pred = myNet(data_s)
        criteon = nn.CrossEntropyLoss()
        # compute loss
        loss_e = criteon(s_pred.cuda(), label_s_e.cuda())
        loss_e.backward()
        optimizer.step()
        optimizer.zero_grad()
        e_output = torch.argmax(s_pred, dim=1)
        correct += torch.eq(e_output, label_s_e).float().sum().item()
    train_acc = correct / (len(source_loader.dataset)) * 100
    train_accuracy.append(train_acc)
    e_rror.append(loss_e.item())
    item_pr = 'Train Epoch: [{}/{}], loss: {:.4f}, Epoch{}_TrainAcc: {:.4f}' \
        .format(epoch, cfg.n_epochs, loss_e.item(), epoch, train_acc)
    print(item_pr)
    logging.info(item_pr)


if __name__ == '__main__':
    # ============================settings=======================================
    cfg_mode = 'deapA'
    # cfg_mode = 'deapV'
    # cfg_mode = 'seed'
    cfg = Config().get_args('./config.yaml')[cfg_mode]
    cfg = DotDict(cfg)
    print(cfg_mode)
    log_dir = cfg.log_path[:cfg.log_path.rfind('/')]
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s %(levelname)s: %(message)s',
        filename=cfg.log_path,
        filemode=cfg.file_mode
    )
    if torch.cuda.is_available():
        print("=" * 120 + '\n' + "cuda is available!" + '\n' + "=" * 120)
    # ===============================start train========================================
    for m in range(1, cfg.exp_times + 1):
        print("experiment:", m)
        logging.info("experiment: " + str(m))
        subject_ACC = []
        for target_sub_id in range(cfg.num_subs):
            print(f"train: {target_sub_id + 1}")
            logging.info(f"train: {target_sub_id + 1}")
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.fastest = True

            index = np.arange(0, cfg.num_subs, 1).tolist()
            del index[target_sub_id]
            print(index)

            source_loader, target_loader = dataloaders.get_seed_and_deap(cfg, index, target_sub_id)
            myNet = model.SNN_Model(cfg).cuda()
            optimizer = optim.Adam(myNet.parameters(), lr=cfg.lr)
            # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            train_accuracy = []
            test_accuracy = []
            e_rror = []
            print('train begin')
            for epoch in range(1, cfg.n_epochs + 1):
                train(myNet, source_loader)
                test_acc = tst(myNet, target_loader)
                test_info = 'Test acc Epoch{}: {:.4f}'.format(epoch, test_acc)
                print(test_info)
                logging.info(test_info)
                test_accuracy.append(test_acc)
            last_ACC = sum(test_accuracy[-3:]) / 3
            subject_ACC.append(last_ACC)

            save_dict = {"train_accuracy": train_accuracy, "test_accuracy": test_accuracy, "loss_error": e_rror}
            path = cfg.output_path + 'experiment-' + str(m) + '/'
            if not os.path.exists(path): os.makedirs(path)
            save_csv(path + f"/target{target_sub_id}.csv", **save_dict)

        subject_ACC = np.array(subject_ACC)
        print(f"the last mean acc is {subject_ACC.mean()}%,the last std acc is {subject_ACC.std()}%\n")
        logging.info(f"the last mean acc is {subject_ACC.mean()}%,the last std acc is {subject_ACC.std()}%\n")
