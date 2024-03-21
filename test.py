import os
from utils import process, te_plot, write_log, calculate_err, make_model
from config import *

if __name__ == '__main__':
    os.makedirs(f'log/{r_name}', exist_ok=True)
    txt_list = []
    all_data = data[:, output_col]
    if standard_mode is not None:
        mn = [[train_mn[0][i], train_mn[1][i]] for i in range(len(train_mn[0]))]
        mn = mn[output_col]

    Dte = process(test_data, 1, False, num_workers, input_len, output_len, output_col)
    model = make_model(input_size, Cfg, device, gpus, f'models/model-{r_name}.pth')
    write_log(f'model: {r_name}', txt_list)
    y_re, y_real = te_plot('log', r_name, model, test_num, t, Dte, all_data, device, output_len,
                           input_len, txt_list, mn, standard_mode)

    MSE, RMSE, MAE, MAPE, SMAPE, R2 = calculate_err(y_re, y_real)

    write_log(f'MSE: {MSE}', txt_list)
    write_log(f'RMSE: {RMSE}', txt_list)
    write_log(f'MAE: {MAE}', txt_list)
    write_log(f'R2: {R2}', txt_list)
    write_log(f'MAPE: {MAPE} %', txt_list)
    write_log(f'SMAPE: {SMAPE} %', txt_list)

    content = ''
    for txt in txt_list:
        content += txt
    with open(f'log/{r_name}/log-{r_name}.txt', 'w+', encoding='utf8') as f:
        f.write(content)
