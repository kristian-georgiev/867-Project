# python main.py --model_training train_new --dataset quickdraw --meta_learner maml
# for index in 1 2 3 4 5 6 7 8 9 10
# do
#     python main.py --model_training pretrained --index ${index} --dataset quickdraw --meta_learner maml
# done


# python main.py --model_training train_new --dataset quickdraw --meta_learner sgd
for index in 1 2 3 4 5 6 7 8 9 10
do
    python main.py --model_training pretrained --index ${index} --dataset quickdraw --meta_learner anil
done

# python main.py --model_training train_new --dataset omniglot --meta_learner sgd
# for index in 1 2 3 4 5 6 7 8 9 10
# do
#     python main.py --model_training pretrained --index ${index} --dataset omniglot --meta_learner sgd
# done

# python main.py --model_training train_new --dataset omniglot --meta_learner maml
# for index in 1 2 3 4 5 6 7 8 9 10
# do
#     python main.py --model_training pretrained --index ${index} --dataset omniglot --meta_learner maml
# done