# 基于LSTM的时序预测

### 本项目基于LSTM CNN+LSTM Seq2Seq 构建时序预测模型，输入为多变量多步长二维序列，输出为单变量多步长一维序列

## 一、 环境部署

首先需安装 python>=3.10.2，然后安装torch>=2.1.1,torchaudio>=2.1.1 torchvision>=0.16.1

在有nvidia服务的设备上，使用以下命令安装

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

在没有nvidia服务的设备上，使用以下命令安装

```bash
pip3 install torch torchvision torchaudio
```

安装后可使用以下命令依次查看torch，cuda版本

```bash
python -c "import torch;print(torch.__version__);print(torch.version.cuda)"
```

安装其他环境依赖

```bash
pip install -r requirements.txt
```



## 二、数据准备

请将预处理得到的二维时间序列矩阵 data 和时间数据 t 写入numpy矩阵中，保存为npz格式

其中二维时间序列矩阵的第一维是不同变量取值，第二维是不同时间取值

```python
import numpy as np
data=[[x1,y1,z1],
      [x2,y2,z2],
      [x3,y3,z3],
         ...
      [xn,yn,zn]]

t=[t1,t2,t3 ... tn]
np.savez('data/static.npz', t=t, data=data, allow_pickle=True)
```





## 三、模型训练

按照注释修改并保存好Cfg.yaml配置文件

```yaml
base:
  model: LSTM                 #选择模型 LSTM CNN_LSTM Seq2Seq
  input_len: 30              #输入长度
  output_col: 1               #输出的列序号
  output_name: 开盘         #输出的名称
  output_len: 7              #输出长度
  standard_mode: 2            #标准化模式 0:无标准化 1:Z-score标准化 2:最大最小标准化


train:
  test_rate: 0.1            #测试集比例
  val_rate: 0.2             #验证集比例
  batch_size: 128            #批大小
  epochs: 200                #训练轮数
  hidden_size: 256          #隐藏层大小
  learn_rate: 0.001         #学习率
  num_layers: 1             #LSTM层数
  step_size: 1              #学习率衰减步长
  gamma: 0.95               #学习率衰减率
```

根据数据和配置文件路径开启训练：

```bash
python train.py --data data/static.npz --config Cfg.yaml
```

训练后的模型、训练过程损失图保存在results文件夹中



## 四、模型测试

将训练好模型移到models文件夹中，开启测试

```bash
python test.py --data data/static.npz --config Cfg.yaml
```

测试结果保存在log文件夹中


