import torchmeta
from torchmeta.utils.data import BatchMetaDataLoader

def dataloader(dataset):
    dataloader = torchmeta.datasets
    dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)