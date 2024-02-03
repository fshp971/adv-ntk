cd $1


for i in {1..3}
do

python train_nn.py \
    --arch rn34 \
    --criterion mse \
    --activation Relu \
    --dataset cifar10 \
    --trainset-size 12000 \
    --batch-size 128 \
    --train-steps 20000 \
    --optim sgd \
    --lr 0.01 \
    --lr-decay-rate 0.1 \
    --lr-decay-freq 8000 \
    --weight-decay 5e-4 \
    --momentum 0.9 \
    --pgd-radius 8 \
    --pgd-steps 10 \
    --pgd-step-size 1.6 \
    --pgd-random-start \
    --pgd-norm-type l-infty \
    --report-freq 1000 \
    --save-freq 100000 \
    --data-dir ./data \
    --save-dir ./exp_data/c10/rn34/at/r8/$i \
    --save-name train

done
