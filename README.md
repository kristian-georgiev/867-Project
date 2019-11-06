# 867-Project

To run this code, you need to install the [higher](https://github.com/facebookresearch/higher) library first. You should run `classification.py` to do experiments. In `meta_ops.py` we collect different meta learning algorithms. 


# To run (updated):

- Clone this repository.
- Install `higher` with 
    ```
    git clone git@github.com:facebookresearch/higher.git
    cd higher
    pip install .
    ```
- Run `pip install -r requirements.txt`
- Run `chmod +x run.sh`
- Run `./run.sh`

If you need to make any configuration adjustments (i.e. gpu support, model directories, etc.), edit `config.yaml`.

If you want to change the dataset, meta-learner (maml, anil), or make any hyperparameter changes, edit `hparams.yaml`.