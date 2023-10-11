cd $1


deps=("3" "4" "5" "8" "10")

for i in {1..3}
do
    for dep in ${deps[@]}
    do

python eval_ntk.py \
    --arch-nt mlp-x \
    --arch-depth ${dep} \
    --W-std 1.76 \
    --b-std 0.18 \
    --criterion mse \
    --activation Relu \
    --dataset cifar10 \
    --trainset-size 12000 \
    --ntk-batch-size 12000 \
    --val-batch-size 128 \
    --pgd-radius 8 \
    --pgd-steps 10 \
    --pgd-step-size 1.6 \
    --pgd-random-start \
    --pgd-norm-type l-infty \
    --data-dir ./data \
    --save-dir ./exp_data/c10/mlp/ntk/dep-${dep}/r8 \
    --save-name eval-$i

    done
done
