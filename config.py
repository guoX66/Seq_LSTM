import argparse
import numpy as np
from utils import read_cfg, ini_env, standardize, cal_mn

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/static.npz')
parser.add_argument('--config', type=str, default='Cfg.yaml')
args = parser.parse_args()
ba_cfg, tr_cfg, Cfg = read_cfg(args.config)
model_name = ba_cfg['model']
input_len = ba_cfg['input_len']
output_col = ba_cfg['output_col']
output_name = ba_cfg['output_name']
output_len = ba_cfg['output_len']
standard_mode = ba_cfg['standard_mode']

test_rate = tr_cfg['test_rate']
val_rate = tr_cfg['val_rate']
max_epochs = tr_cfg['epochs']
batch_size = tr_cfg['batch_size']
hidden_size = tr_cfg['hidden_size']
num_layers = tr_cfg['num_layers']
learn_rate = tr_cfg['learn_rate']
step_size = tr_cfg['step_size']
gamma = tr_cfg['gamma']

device, gpus, num_workers = ini_env()
npfile = np.load(args.data, allow_pickle=True)
data = npfile['data'].astype(np.float64)
t = npfile['t']
if data.ndim != 2:
    raise ValueError('Data should be 2D')

input_size = data.shape[1]
train_mn = cal_mn(data[:int(len(data) * (1 - val_rate - test_rate))], standard_mode)  # 计算训练集归一化参数
data_nor = standardize(data, standard_mode, train_mn)  # 用训练集归一化参数归一化整个数据集

test_num = int(int(len(data) * (1 - test_rate)))
train_data = data_nor[:int(len(data) * (1 - val_rate - test_rate))]
val_data = data_nor[int(len(data) * (1 - val_rate - test_rate)):test_num]
test_data = data_nor[test_num:]

r_name = f'{model_name}-{output_name}'
