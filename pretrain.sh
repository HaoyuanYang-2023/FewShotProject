

TRAIN_ROOT=/media/ace2nou/F/miniImageNet/train
VAL_ROOT=/media/ace2nou/F/miniImageNet/test

echo '==pretrain=='
python pretrain.py --dataset miniImageNet --train_root $TRAIN_ROOT --val_root $VAL_ROOT --epochs 160 --milestones 80 120 140 \
                   --reduced_dim 640 --dropout_rate 0.8 --val meta --rand_aug --label_smooth --smooth 0.2 --exp ls-ra
