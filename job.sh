# for index in 1 2 3 4 5 6 7 8 9 10
# do
#     python main.py --model_training pretrained --index ${index} --dataset quickdraw
# done

# for index in 1 2 3 4 5 6 7 8 9 10
# do
#     python main.py --model_training pretrained --index ${index} --dataset omniglot
# done

for index in 1 2 3 4 5 6 7 8 9 10
do
    python main.py --model_training pretrained --index ${index} --dataset omniglot --meta_learner sgd
done

# for index in 1 2 3 4 5 6 7 8 9 10
# do
#     python main.py --model_training pretrained --index ${index} --dataset quickdraw --meta_learner sgd
# done