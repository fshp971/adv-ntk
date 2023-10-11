cd $1


deps=("3" "4" "5" "8" "10")

for i in {1..3}
do
    for dep in ${deps[@]}
    do

python train_advntk.py \
    --arch-nt cnn-x \
    --arch-depth ${dep} \
    --W-std 1.76 \
    --b-std 0.18 \
    --criterion mse \
    --activation Relu \
    --dataset cifar10 \
    --trainset-size 12000 \
    --val-num 2000 \
    --ntk-batch-size 500 \
    --val-batch-size 128 \
    --gd-steps 50 \
    --gd-normalize \
    --gd-weight-decay 0 \
    --gd-lr 0.1 \
    --gd-lr-decay-rate 1 \
    --gd-lr-decay-freq 1000 \
    --pgd-radius 8 \
    --pgd-steps 10 \
    --pgd-step-size 1.6 \
    --pgd-random-start \
    --pgd-norm-type l-infty \
    --eval-freq 25 \
    --save-freq 1000 \
    --data-dir ./data \
    --save-dir ./exp_data/c10/cnn/advntk/dep-${dep}/r8/$i \
    --save-name train

    done
done
