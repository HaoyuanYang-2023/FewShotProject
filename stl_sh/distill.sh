cd ../

TRAIN_ROOT=/media/
VAL_ROOT=/media/
PRETRAIN_MODEL_PATH=./runs/miniImageNet/MPNCOVResNet12_pretrain_dim196/last_model.pth
echo '==born1=='
nohup python distillation.py --dataset miniImageNet --train_root $TRAIN_ROOT --val_root $VAL_ROOT --born 1 --epochs 170 --milestones [100,150] --pretrain_model_path $PRETRAIN_MODEL_PATH --exp dim196 >./logs/MPNCOV_dis_dim196_born1.log 2>&1 &

PRETRAIN_MODEL_PATH_1=./runs/miniImageNet/MPNCOVResNet12_distillation_born1_dim196/last_model.pth
echo '==born2=='
nohup python distillation.py --dataset miniImageNet --train_root $TRAIN_ROOT --val_root $VAL_ROOT --born 2 --epochs 170 --milestones [100,150] --pretrain_model_path $PRETRAIN_MODEL_PATH_1 --exp dim196 >./logs/MPNCOV_dis_dim196_born1.log 2>&1 &

PRETRAIN_MODEL_PATH_2=./runs/miniImageNet/MPNCOVResNet12_distillation_born2_dim196/last_model.pth
echo '==born3=='
nohup python distillation.py --dataset miniImageNet --train_root $TRAIN_ROOT --val_root $VAL_ROOT --born 3 --epochs 170 --milestones [100,150] --pretrain_model_path $PRETRAIN_MODEL_PATH_2 --exp dim196 >./logs/MPNCOV_dis_dim196_born1.log 2>&1 &