import torch
import torch.distributed
import torchvision
import argparse
import deepspeed
from model import FashionModel, img_transform

## 命令行参数 deepspeed ds_eval.py --deepspeed --deepspeed_config ds_config.json
########################### 推理准备，所有子进程都要执行 ###########################
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
parser = deepspeed.add_config_arguments(parser)
cmd_args = parser.parse_args() # deepspeed命令行参数

model = FashionModel().cuda() # 原始模型
model, _, _, _ = deepspeed.initialize(args=cmd_args, model=model, model_parameters=model.parameters()) # deepspeed分布式模型
model.load_checkpoint('./checkpoints') # 加载参数

model.eval() # 分布式推理模式

###################### 只有主控进程带头做这些动作 #################################
if torch.distributed.get_rank()==0:
    dataset = torchvision.datasets.FashionMNIST(root='./dataset', download=True, transform=img_transform) # 衣服数据集
    batch_x = torch.stack([dataset[0][0], dataset[1][0]]).cuda()

    outputs = model(batch_x) # 分布式推理
    print('分布式推理：', outputs.cpu().argmax(dim=1), [dataset[0][1], dataset[1][1]])

    #################################### 模型转成torch单体 ##################################

    torch.save(model.module.state_dict(), 'model.pt') # 保存为普通torch模型参数

    model = FashionModel().cuda() # 加载torch模型
    model.load_state_dict(torch.load('model.pt'))

    model.eval() # 单体推理
    outputs = model(batch_x)
    print('单体推理', outputs.cpu().argmax(dim=1), [dataset[0][1], dataset[1][1]])

