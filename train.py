import shutil
import time
import os
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from utils import process, val_plot, bar, make_model
from config import *

base_path = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    print(f'Making dataset...')
    Dtr = process(train_data, batch_size, True, num_workers, input_len, output_len, output_col)
    DVa = process(val_data, batch_size, True, num_workers, input_len, output_len, output_col)
    print('Done!')
    model = make_model(input_size, Cfg, device, gpus)
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    loss_fn = nn.MSELoss()
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    min_val_loss = np.Inf
    loss_list = []
    min_lost = np.Inf

    date_time = time.strftime('%Y_%m_%d-%Hh_%Mm', time.localtime())
    filename = r_name + '-' + str(date_time)
    file_path = os.path.join(base_path, 'results', filename)
    os.makedirs(file_path, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    print(f'{r_name} training...')
    start_time = time.perf_counter()
    for epoch in range(max_epochs):
        train_loss = []
        model.train()
        for (seq, label) in Dtr:
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            # print(label.shape, y_pred.shape)
            loss = loss_fn(y_pred, label)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        total_val_loss = 0
        with torch.no_grad():  # 验证数据集时禁止反向传播优化权重
            for seq, label in DVa:
                seq = seq.to(device)
                label = label.to(device)
                outputs = model(seq)
                loss = loss_fn(outputs, label)
                total_val_loss = total_val_loss + loss.item()
            loss_list.append(total_val_loss)
            if total_val_loss < min_val_loss:
                min_val_loss = total_val_loss
                m_epoch = epoch
                torch.save(model.state_dict(), f"{file_path}/model-{r_name}.pth")  # 保存最好的模型
        bar(epoch + 1, max_epochs, start_time, '训练进度', train=True, loss=total_val_loss)
    shutil.copy(f"{file_path}/model-{r_name}.pth", f"models/model-{r_name}.pth")
    print()
    print(f'本次训练损失最小的epoch为{m_epoch},最小损失为{min_val_loss}')
    val_plot(m_epoch, loss_list, f"{file_path}/model-{r_name}-loss", f'{r_name}-loss')
