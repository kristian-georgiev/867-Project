import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

from sklearn.decomposition import PCA


def pca_directions(weights_accross_training):
    # TODO: implement flatten()
    flat_weight_tensor = flatten(weights_accross_training)
    flat_weight_np = flat_weight_tensor.numpy()
    pca = PCA(n_components=2)
    pca.fit(flat_weight_np)
    return pca.components_

def plot_loss_landscape(directions, test_dataset, model):
    pass

def plot_progress(log):
    # Generally you should pull your plotting code out of your training
    # script but we are doing it here for brevity.
    df = pd.DataFrame(log)

    fig, ax = plt.subplots(figsize=(6, 4))
    train_df = df[df['mode'] == 'train']
    test_df = df[df['mode'] == 'test']
    ax.plot(train_df['epoch'], train_df['acc'], label='Train')
    ax.plot(test_df['epoch'], test_df['acc'], label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(70, 100)
    fig.legend(ncol=2, loc='lower right')
    fig.tight_layout()
    fname = 'maml-accs.png'
    print(f'--- Plotting accuracy to {fname}')
    fig.savefig(fname)
    plt.close(fig)


def flatten(weights_list):
    for w in weights_list:
        # flatten state_dict into tensor
        pass
    return weights_list