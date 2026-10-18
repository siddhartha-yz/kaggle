"""Task-local PyTabKit adapter: CPU initialization/data, CUDA minibatches.

No installed files are changed. Initialization and shuffle use CPU RNG, so this
is a declared execution variant, not a claim of identical GPU-training weights.
"""
from contextlib import contextmanager
import pytabkit.models.training.lightning_modules as lm
import pytabkit.models.alg_interfaces.nn_interfaces as ni
from pytabkit.models.training.nn_creator import NNCreator


class CPUDataCreator(NNCreator):
    def create_model(self, ds, idxs_list):
        device = self.device_info
        self.device_info = 'cpu'
        try:
            model = super().create_model(ds.to('cpu'), idxs_list)
        finally:
            self.device_info = device
        assert all(p.device.type == 'cpu' for p in self.static_model.parameters())
        return model.to(device)

    def create_dataloaders(self, ds):
        device = self.device_info
        self.device_info = 'cpu'
        self.train_idxs = self.train_idxs.cpu()
        if self.val_idxs is not None:
            self.val_idxs = self.val_idxs.cpu()
        try:
            train, valid = super().create_dataloaders(ds.to('cpu'))
        finally:
            self.device_info = device
        assert str(train.ds.device) == 'cpu'
        train.output_device = device
        # The research runner supplies no outer validation data to fit().
        assert valid is None, 'This audited adapter currently supports fixed-epoch fits only'
        return train, valid


class StreamingTabNNModule(lm.TabNNModule):
    def __init__(self, *args, **kwargs):
        original = lm.NNCreator
        lm.NNCreator = CPUDataCreator
        try:
            super().__init__(*args, **kwargs)
        finally:
            lm.NNCreator = original

    def get_predict_dataloader(self, ds):
        loader = super().get_predict_dataloader(ds.to('cpu'))
        assert str(loader.ds.device) == 'cpu'
        loader.output_device = self.creator.device_info
        return loader


@contextmanager
def streaming_backend():
    original = ni.TabNNModule
    ni.TabNNModule = StreamingTabNNModule
    try:
        yield
    finally:
        ni.TabNNModule = original
