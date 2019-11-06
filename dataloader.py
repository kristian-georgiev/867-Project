# import torchmeta
# from torchmeta.utils.data import BatchMetaDataLoader


# TODO: incorporate torchmeta or meta-dataset

def dataloader(hparams):
    if hparams.dataset == "omniglot":
        from support.omniglot_loaders import OmniglotNShot
        return OmniglotNShot(
                '/tmp/omniglot-data',
                batchsz=hparams.meta_batch_size,
                n_way=hparams.n_way,
                k_shot=hparams.k_support,
                k_query=hparams.k_query,
                imgsz=28,
                device=hparams.device,
                )
    elif hparams.dataset == "quickdraw":
        from support.quickdraw_loaders import QuickdrawNShot
        return QuickdrawNShot(
            './support/data/QuickDrawData.pkl',
                batchsz=hparams.meta_batch_size,
                n_way=hparams.n_way,
                k_shot=hparams.k_support,
                k_query=hparams.k_query,
                imgsz=28,
                device=hparams.device,
                )
