cd ../

TRAIN_ROOT=/media/ace2nou/F/miniImageNet/train
VAL_ROOT=/media/ace2nou/F/miniImageNet/test
# PRETRAIN_MODEL_PATH=./runs/miniImageNet/MPNCOVResNet12_pretrain_dim196/last_model.pth
PRETRAIN_MODEL_PATH=/home/ace2nou/yanghaoyuan/FewShotProject/pretrained/MPNCOVResNet12_pretrain_dim192_drop0.5_last_model.pth
echo '==born1=='
python distillation.py --dataset miniImageNet --train_root $TRAIN_ROOT --val_root $VAL_ROOT --born 1 --epochs 170 --milestones [100,150] --pretrain_model_path $PRETRAIN_MODEL_PATH --reduced_dim 192 --dropout_rate 0.5 --exp dim196 --val last # >./logs/MPNCOV_dis_dim196_born1.log 2>&1 &

PRETRAIN_MODEL_PATH_1=./runs/miniImageNet/MPNCOVResNet12_distillation_born1_dim196/last_model.pth
echo '==born2=='
python distillation.py --dataset miniImageNet --train_root $TRAIN_ROOT --val_root $VAL_ROOT --born 2 --epochs 170 --milestones [100,150] --pretrain_model_path $PRETRAIN_MODEL_PATH_1 --reduced_dim 196 --dropout_rate 0.5 --exp dim196 --val last # >./logs/MPNCOV_dis_dim196_born1.log 2>&1 &

PRETRAIN_MODEL_PATH_2=./runs/miniImageNet/MPNCOVResNet12_distillation_born2_dim196/last_model.pth
echo '==born3=='
python distillation.py --dataset miniImageNet --train_root $TRAIN_ROOT --val_root $VAL_ROOT --born 3 --epochs 170 --milestones [100,150] --pretrain_model_path $PRETRAIN_MODEL_PATH_2 --reduced_dim 196 --dropout_rate 0.5 --exp dim196 --val last # >./logs/MPNCOV_dis_dim196_born1.log 2>&1 &