

TRAIN_ROOT=/root/autodl-tmp/miniImageNet/train
VAL_ROOT=/root/autodl-tmp/miniImageNet/test

echo '==pretrain=='
python pretrain.py --dataset miniImageNet --train_root $TRAIN_ROOT --val_root $VAL_ROOT --epochs 160 --milestones 80 120 140 \
--reduced_dim 640 --dropout_rate 0.8 --val meta  --exp ema_smooth
