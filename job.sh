# python main.py --model_training train_new --dataset quickdraw --meta_learner maml
# for index in 1 2 3 4 5 6 7 8 9 10
# do
#     python main.py --model_training pretrained --index ${index} --dataset quickdraw --meta_learner maml --fix_head
# done


# python main.py --model_training train_new --dataset quickdraw --meta_learner sgd
# for index in 1 2 3 4 5 6 7 8 9 10
# do
#     python main.py --model_training pretrained --index ${index} --dataset quickdraw --meta_learner anil
# done

# python main.py --model_training train_new --dataset omniglot --meta_learner sgd
# for index in 1 2 3 4 5 6 7 8 9 10
# do
#     python main.py --model_training pretrained --index ${index} --dataset omniglot --meta_learner sgd
# done

# python main.py --model_training train_new --dataset omniglot --meta_learner maml
# for index in 1 2 3 4 5 6 7 8 9 10
# do
#     python main.py --model_training pretrained --index ${index} --dataset omniglot --meta_learner maml --fix_head
# done

for n_inner_iter in 0 1 2 3 4 5 8 10
do
    python main.py --model_training pretrained --index 7 --n_inner_iter ${n_inner_iter} --dataset omniglot --meta_learner maml
done
