cd ../

TRAIN_ROOT=F:/miniImageNet/train
VAL_ROOT=F:/miniImageNet/test

echo '==pretrain=='
python pretrain.py --dataset miniImageNet --train_root $TRAIN_ROOT --val_root $VAL_ROOT --epochs 170 --milestones [100,150] --reduced_dim 196 --dropout_rate 0.5 --exp dim196 --val last
