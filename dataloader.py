# import torchmeta
# from torchmeta.utils.data import BatchMetaDataLoader

from support.omniglot_loaders import OmniglotNShot

# TODO: incorporate torchmeta or meta-dataset

def dataloader(hparams):
    if hparams.dataset == "omniglot":
        return OmniglotNShot(
                '/tmp/omniglot-data',
                batchsz=hparams.meta_batch_size,
                n_way=hparams.n_way,
                k_shot=hparams.k_support,
                k_query=hparams.k_query,
                imgsz=28,
                device=hparams.device,
            )