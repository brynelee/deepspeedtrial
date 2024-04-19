import torch
import torchvision
import argparse
import deepspeed
from torchvision import transforms
from model import FashionModel, img_transform

## 命令行参数 deepspeed ds_train.py --epoch 5 --deepspeed --deepspeed_config ds_config.json
########################### 训练逻辑，每个子进程都要执行完整代码，彼此共同协商训练 ###########################
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
parser.add_argument('--epoch', type=int, default=-1, help='epoch')
parser = deepspeed.add_config_arguments(parser)
cmd_args = parser.parse_args() # deepspeed命令行参数

dataset = torchvision.datasets.FashionMNIST(root='./dataset', download=True, transform=img_transform) # 衣服数据集
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=1, shuffle=True)

model = FashionModel().cuda() # 原始模型
model, _, _, _ = deepspeed.initialize(args=cmd_args, model=model, model_parameters=model.parameters()) # deepspeed分布式模型
loss_fn = torch.nn.CrossEntropyLoss().cuda()

for epoch in range(cmd_args.epoch):
    for x, y in dataloader:
        x, y = x.cuda(), y.cuda()
        output = model(x)
        loss = loss_fn(output, y)
        model.backward(loss) # 走deepspeed风格的backward
        model.step()
    print('epoch {} done...'.format(epoch))
    model.save_checkpoint('./checkpoints')


