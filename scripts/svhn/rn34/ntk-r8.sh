cd $1


for i in {1..3}
do

python eval_ntk.py \
    --arch-nt rn34 \
    --W-std 1.76 \
    --b-std 0.18 \
    --criterion mse \
    --activation Relu \
    --dataset svhn \
    --trainset-size 12000 \
    --ntk-batch-size 250 \
    --val-batch-size 128 \
    --pgd-radius 8 \
    --pgd-steps 10 \
    --pgd-step-size 1.6 \
    --pgd-random-start \
    --pgd-norm-type l-infty \
    --data-dir ./data \
    --save-dir ./exp_data/svhn/rn34/ntk/r8 \
    --save-name eval-$i

done
