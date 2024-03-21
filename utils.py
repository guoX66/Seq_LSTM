import os
import platform
import torch
import yaml
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
from LSTM_models import LSTM, CNN_LSTM, Seq2Seq


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def select_model(model_name, input_size, hidden_size, num_layers, output_len, batch_size, device):
    if model_name == 'LSTM':
        model = LSTM(input_size, hidden_size, num_layers, output_len, batch_size, device)
    elif model_name == 'CNN_LSTM':
        model = CNN_LSTM(input_size, hidden_size, num_layers, output_len, batch_size, device)
    elif model_name == 'Seq2Seq':
        model = Seq2Seq(input_size, hidden_size, num_layers, output_len, batch_size, device)
    else:
        raise ValueError('model name error')
    return model


def make_model(input_size, Cfg, device, gpus, path=None):
    ba_cfg, tr_cfg = Cfg['base'], Cfg['train']
    model_name, output_len = ba_cfg['model'], ba_cfg['output_len']
    hidden_size, num_layers, batch_size = tr_cfg['hidden_size'], tr_cfg['num_layers'], tr_cfg['batch_size']
    model = select_model(model_name, input_size, hidden_size, num_layers, output_len, batch_size, device)
    model = model.to(device)
    if path is not None:
        model.load_state_dict(torch.load(path, map_location=device))
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
    model = model.to(device)
    if len(gpus) > 1:
        model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])  # DP
    return model


def write_log(log, txt_list):
    print(log)
    txt_list.append(log + '\r\n')


def process(data, batch_size, shuffle, num_workers, input_len, output_len, output_col):
    data = data.tolist()
    seq = []
    for i in range(len(data) - input_len - output_len):
        train_seq = []
        train_label = []
        for j in range(i, i + input_len):
            x = data[j]
            train_seq.append(x)
        for j in range(i + input_len, i + input_len + output_len):
            x = data[j][output_col]
            train_label.append(x)
        train_seq = torch.FloatTensor(train_seq)
        # train_label = torch.FloatTensor(train_label).view(-1)

        train_label = torch.FloatTensor(train_label)
        # print(train_seq.shape, train_label.shape)

        seq.append((train_seq, train_label))

    seq = MyDataset(seq)
    seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    return seq


def compare_plot(fold_name, title, t_pre, y_pre, t_real, y_real):
    figure(figsize=(12.8, 9.6))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(t_pre, y_pre, color='blue', label='predict')
    plt.plot(t_real, y_real, color='red', label='real')
    plt.title(f'{title}', fontsize=20)
    plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 30, 'fontsize', 10)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig(f'{fold_name}/{title}.png')


def te_plot(fold_name, title, lstm_model, test_num, t_all, Dte, data, device, pred_size, interval, txt_list, mn,
            standard_mode):
    ts = test_num + interval
    te = test_num + interval + pred_size + len(Dte) - 1
    t_test = t_all[ts:te]
    y_pred_list = np.zeros((len(Dte), len(Dte) + pred_size - 1), dtype=float)
    count = 0
    lstm_model.eval()
    st = time.perf_counter()
    for seq, _ in Dte:
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = lstm_model(seq)
            y_pred = y_pred.cpu().detach().numpy()
            y_pred_list[count, count:count + len(y_pred[0])] = y_pred[0]

            # y_pred_list += y_pred.tolist()
        count += 1

    et = time.perf_counter()
    wt = str(round(1000 * (et - st) / len(Dte), 3))
    write_log('time per prediction: ' + wt + ' ms', txt_list)
    y_re = []
    for i in range(len(Dte) + pred_size - 1):
        col = y_pred_list[:, i]
        not_zero = np.where(col != 0)
        r_col = col[not_zero]
        # 按照与首个输入值间的距离计算权重
        # l = np.max(not_zero) + 1
        # dis = np.array([i + 1 for i in range(l) if col[i]])
        # dis = dis / sum(dis)
        # va = sum(r_col * dis)
        # 按照相对距离计算权重
        dis = np.arange(len(r_col)) + 1
        dis = dis / sum(dis)
        va = sum(r_col * dis)

        #va = np.mean(r_col)
        y_re.append(va)
    y_re = np.array(y_re)
    if standard_mode == 1:
        y_re = y_re * mn[1] + mn[0]
    elif standard_mode == 2:
        y_re = y_re * (mn[0] - mn[1]) + mn[1]
    y_real = data[ts:te]
    compare_plot(f'{fold_name}/{title}', f'{title}-all', t_test, y_re, t_all, data)
    compare_plot(f'{fold_name}/{title}', f'{title}-test', t_test, y_re, t_test, y_real)
    return y_re, y_real


def val_plot(m_epoch, loss_list, img_path, title):
    min_val_loss = min(loss_list)
    epochs = len(loss_list)
    figure(figsize=(12.8, 9.6))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(loss_list, color='red', label='LOSS')
    plt.scatter(m_epoch, min_val_loss, color='blue', s=80)
    plt.scatter(0.73 * epochs, 0.95 * max(loss_list), color='blue', s=80)
    min_val_loss = '%.6f' % min_val_loss
    text = f'min loss:{min_val_loss}'
    plt.text(0.75 * epochs, 0.95 * max(loss_list), text, ha='left', va='center', size=20)
    plt.title(title, fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim((0, max(loss_list)))
    plt.savefig(f'{img_path}.png')


def bar(i, t, start, des, train=True, loss=0):
    l = 50
    f_p = i / t
    n_p = (t - i) / t
    finsh = "▓" * int(f_p * l)
    need_do = "-" * int(n_p * l)
    progress = f_p * 100
    dur = time.perf_counter() - start
    if train:
        proc = "\r{}({}/{}轮):{:^3.2f}%[{}->{}] 用时:{:.2f}s 验证集上损失:{:.3f} ".format(des, i, t,
                                                                                          progress,
                                                                                          finsh,
                                                                                          need_do, dur,
                                                                                          loss)
    else:
        proc = "\r{}:{:^3.2f}%[{}->{}] 用时:{:.2f}s".format(des, progress, finsh, need_do, dur)
    print(proc, end="")


def calculate_err(y_re, y_real):
    MSE = np.sum((y_re - y_real) ** 2) / len(y_real)
    RMSE = np.sqrt(MSE)
    MAE = np.sum(np.abs(y_re - y_real)) / len(y_real)
    MAPE = np.sum(np.abs((y_re - y_real) / y_real)) / len(y_real) * 100
    S = np.abs(y_re - y_real) / ((np.abs(y_re) + np.abs(y_real)) / 2)
    SMAPE = np.sum(S) / len(y_real) * 100
    av_y = sum(y_real) / len(y_real)
    R2 = 1 - np.sum((y_re - y_real) ** 2) / np.sum((av_y - y_real) ** 2)
    return MSE, RMSE, MAE, MAPE, SMAPE, R2


def read_cfg(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            Cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        return Cfg['base'], Cfg['train'], Cfg
    else:
        raise FileNotFoundError(f'{path} not found')


def cal_mn(data, mode):  # 计算训练集均值和标准差或最大最小值
    if mode == 0:
        return None
    elif mode == 1:
        m1, n1 = np.mean(data, axis=0), np.std(data, axis=0)
        return [m1, n1]
    elif mode == 2:
        m1, n1 = np.max(data, axis=0), np.min(data, axis=0)
        return [m1, n1]
    else:
        raise ValueError('Standard mode error')


def standardize(data, mode, train_mn):
    if mode == 0:
        return data
    elif mode == 1:
        m1, n1 = train_mn
        data = (data - m1) / n1
        return data
    elif mode == 2:
        m1, n1 = train_mn
        data = (data - n1) / (m1 - n1)
        return data
    else:
        raise ValueError('Standard mode error')


def ini_env():
    if torch.cuda.is_available():
        gpu_num = torch.cuda.device_count()
        device = torch.device('cuda')
        gpus = [i for i in range(gpu_num)]
        torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    else:
        device = torch.device('cpu')
        gpus = []
    os_name = str(platform.system())
    if os_name == 'Windows':
        num_workers = 0
    else:
        num_workers = 32
    return device, gpus, num_workers
