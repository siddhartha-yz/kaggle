"""Exact membership for canonical string keys, without NumPy object pairwise loops."""
import numpy as np,pandas as pd

def string_support(a,b,sample):
 a=pd.Index(a,dtype=object);b=pd.Index(b,dtype=object);sa=a.take(sample)
 au=a.unique();bu=b.unique();su=sa.unique()
 return dict(full_train_unique=len(au),full_test_unique=len(bu),full_test_rows_unseen_train=int((~b.isin(au)).sum()),equal_size=len(b),equal_train_rows_unseen_test=int((~sa.isin(bu)).sum()),equal_test_rows_unseen_train=int((~b.isin(su)).sum()))
