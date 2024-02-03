cd $1


for i in {1..3}
do

python train_advntk.py \
    --arch-nt rn18 \
    --W-std 1.76 \
    --b-std 0.18 \
    --criterion mse \
    --activation Relu \
    --dataset svhn \
    --trainset-size 12000 \
    --val-num 2000 \
    --ntk-batch-size 250 \
    --val-batch-size 128 \
    --gd-steps 50 \
    --gd-normalize \
    --gd-weight-decay 0 \
    --gd-lr 5e-5 \
    --gd-lr-decay-rate 1 \
    --gd-lr-decay-freq 1000 \
    --pgd-radius 8 \
    --pgd-steps 10 \
    --pgd-step-size 1.6 \
    --pgd-random-start \
    --pgd-norm-type l-infty \
    --eval-freq 50 \
    --save-freq 1000 \
    --data-dir ./data \
    --save-dir ./exp_data/svhn/rn18/advntk/r8/$i \
    --save-name train

done
