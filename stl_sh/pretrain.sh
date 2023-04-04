cd ../

TRAIN_ROOT=/media/
VAL_ROOT=/media/

echo '==pretrain=='
nohup python pretrain.py --dataset miniImageNet --train_root $TRAIN_ROOT --val_root $VAL_ROOT --epochs 170 --milestones [100,150] --reduced_dim 196 --dropout_rate 0.5 --exp dim196 > ./logs/MPNCOV_pretrain_dim196.log 2>&1 &
