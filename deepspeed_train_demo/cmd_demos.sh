# 本地训练
deepspeed ds_train.py --epoch 1 --deepspeed --deepspeed_config ds_config.json

# 本地推理
deepspeed ds_eval.py --deepspeed --deepspeed_config ds_config.json

# 远程训练
deepspeed --hostfile hostfile ds_train.py --epoch 1 --deepspeed --deepspeed_config ds_config.json
