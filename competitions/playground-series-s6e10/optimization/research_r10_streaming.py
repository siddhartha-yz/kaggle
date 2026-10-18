"""CPU-data CUDA adapter with correctly placed validation labels and batches."""
from contextlib import contextmanager
import pytabkit.models.training.lightning_modules as lm
import pytabkit.models.alg_interfaces.nn_interfaces as ni
from pytabkit.models.training.nn_creator import NNCreator
from research_r8_streaming import CPUDataCreator,StreamingTabNNModule

class CPUValidationCreator(CPUDataCreator):
    def create_dataloaders(self,ds):
        device=self.device_info
        self.device_info='cpu'
        self.train_idxs=self.train_idxs.cpu()
        if self.val_idxs is not None:self.val_idxs=self.val_idxs.cpu()
        try:train,valid=NNCreator.create_dataloaders(self,ds.to('cpu'))
        finally:self.device_info=device
        train.output_device=device
        assert str(train.ds.device)=='cpu'
        if valid is not None:
            assert str(valid.val_x_dl.ds.device)=='cpu'
            valid.val_x_dl.output_device=device
            valid.val_y=valid.val_y.to(device)
        return train,valid

class ValidationStreamingModule(StreamingTabNNModule):
    def __init__(self,*args,**kwargs):
        original=lm.NNCreator
        lm.NNCreator=CPUValidationCreator
        try:lm.TabNNModule.__init__(self,*args,**kwargs)
        finally:lm.NNCreator=original

@contextmanager
def validation_streaming_backend():
    original=ni.TabNNModule
    ni.TabNNModule=ValidationStreamingModule
    try:yield
    finally:ni.TabNNModule=original
