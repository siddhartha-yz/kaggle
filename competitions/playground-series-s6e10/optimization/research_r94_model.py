"""Own small tabular RTD: column identities, global attention, no positions or side features."""
import numpy as np
import pandas as pd
import torch
from torch import nn

RATINGS = ['Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling','Checkin service','Cleanliness']
NUM = ['Age','Flight Distance',*RATINGS,'Departure Delay in Minutes','Arrival Delay in Minutes']
CATS = ['Gender','Customer Type','Type of Travel','Class']
FEATURES = NUM + CATS
CONFIG = dict(width=128, layers=3, heads=4, feedforward=256, dropout=0.1)


class Preprocessor:
    def fit(self, frame):
        x = frame[NUM].to_numpy('float64')
        self.mean = np.nanmean(x, axis=0)
        self.scale = np.maximum(np.nanstd(x, axis=0), 1e-6)
        self.vocab = [sorted(frame[c].dropna().astype(str).unique().tolist()) for c in CATS]
        return self

    def state(self):
        return dict(mean=self.mean.tolist(), scale=self.scale.tolist(), vocab=self.vocab)

    def transform(self, frame):
        # Keep original numeric values and missingness together through corruption.
        raw = frame[NUM].to_numpy('float64')
        num = raw.astype('float32')
        assert not np.isinf(num).any()
        assert np.array_equal(np.isnan(raw), np.isnan(num))
        for j in range(len(NUM)):
            assert len(np.unique(raw[np.isfinite(raw[:,j]),j])) == len(np.unique(num[np.isfinite(num[:,j]),j]))
        columns = []
        for name, vocab in zip(CATS, self.vocab):
            mapping = {v:i+2 for i,v in enumerate(vocab)}
            s = frame[name].astype('string')
            columns.append(s.map(mapping).fillna(0).mask(s.isna(),1).to_numpy('int64'))
        return torch.from_numpy(num), torch.from_numpy(np.stack(columns,axis=1))


def corrupt(num, cat, pool_num, pool_cat, selected, donors):
    """Same-column independent donors; target=actual change, not merely selected cell."""
    assert selected.shape == donors.shape == (len(num),21)
    j = torch.arange(17,device=num.device)
    source_num = pool_num[donors[:,:17],j]
    source_cat = pool_cat[donors[:,17:],torch.arange(4,device=cat.device)]
    same_num = (num == source_num) | (torch.isnan(num) & torch.isnan(source_num))
    labels = torch.cat([selected[:,:17] & ~same_num, selected[:,17:] & (cat != source_cat)],dim=1)
    return (torch.where(selected[:,:17],source_num,num),
            torch.where(selected[:,17:],source_cat,cat),labels.float())


class RTDModel(nn.Module):
    def __init__(self, preprocessor):
        super().__init__()
        d = CONFIG['width']
        self.register_buffer('mean',torch.tensor(preprocessor.mean,dtype=torch.float32))
        self.register_buffer('scale',torch.tensor(preprocessor.scale,dtype=torch.float32))
        self.numeric_weight = nn.Parameter(torch.randn(17,d)*0.02)
        self.missing_weight = nn.Parameter(torch.randn(17,d)*0.02)
        self.category = nn.ModuleList([nn.Embedding(len(v)+2,d) for v in preprocessor.vocab])
        self.column = nn.Parameter(torch.randn(21,d)*0.02)
        self.cls = nn.Parameter(torch.randn(1,1,d)*0.02)
        layer = nn.TransformerEncoderLayer(d,CONFIG['heads'],CONFIG['feedforward'],
                    dropout=CONFIG['dropout'],activation='gelu',batch_first=True,norm_first=True)
        self.encoder = nn.TransformerEncoder(layer,CONFIG['layers'],norm=nn.LayerNorm(d),enable_nested_tensor=False)
        self.rtd = nn.Linear(d,1)
        self.classifier = nn.Linear(d,1)

    def forward(self, num, cat, order=None):
        missing = torch.isnan(num)
        values = torch.where(missing,0.0,(num-self.mean)/self.scale)
        numeric = values[:,:,None]*self.numeric_weight + missing[:,:,None]*self.missing_weight
        categorical = torch.stack([emb(cat[:,j]) for j,emb in enumerate(self.category)],dim=1)
        tokens = torch.cat([numeric,categorical],dim=1)+self.column
        if order is not None:
            tokens = tokens[:,order]
        encoded = self.encoder(torch.cat([self.cls.expand(len(num),-1,-1),tokens],dim=1))
        return self.rtd(encoded[:,1:]).squeeze(-1),self.classifier(encoded[:,0]).squeeze(-1)
