"""Reconstruct exact own-training strict-OOF encoder inputs of saved R68 models."""
import numpy as np,pandas as pd
from sklearn.preprocessing import TargetEncoder
from research_features import DistanceFeatures
from research_r4_lean_features import COMBOS

def training_frame(tr,raw,y,aux,fi):
    X=tr.transform(raw.iloc[fi].reset_index(drop=True),aux.iloc[fi].reset_index(drop=True));cross=pd.DataFrame({f'combo{j}':DistanceFeatures._key(raw.iloc[fi].reset_index(drop=True),c) for j,c in enumerate(COMBOS)});encoder=TargetEncoder(**tr.encoder.get_params());te=encoder.fit_transform(cross,y[fi])
    assert encoder.target_mean_==tr.encoder.target_mean_ and all(np.array_equal(a,b) for a,b in zip(encoder.encodings_,tr.encoder.encodings_)) and all(np.array_equal(a,b) for a,b in zip(encoder.categories_,tr.encoder.categories_))
    for j in range(3):X[f'cross_te_{j}']=te[:,j].astype('float32')
    assert X.shape==(len(fi),74) and X.select_dtypes(include='number').shape[1]==52
    return X
