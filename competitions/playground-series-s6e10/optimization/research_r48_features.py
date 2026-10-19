"""Raw21 + validated target-free27 + one safe-original-only SAT teacher =49."""
import numpy as np,pandas as pd,json
from research_r4_common import ROOT,sha
from research_r23_loader import auxiliary

def covariates(name):
    assert name in ['train','test']
    columns=[c for c in pd.read_csv(ROOT/'data/train.csv',nrows=0).columns if c not in ['id','satisfaction']]
    df=pd.read_csv(ROOT/f'data/{name}.csv',usecols=['id',*columns]);raw=df[columns];aux=auxiliary(df.id,test=name=='test')
    source=ROOT/'research_artifacts/original_teacher_v1';record=json.loads((source/'result.json').read_text());assert record['status']=='complete' and record['safe_rows']==129859 and record['excluded_exact_rows']==21 and record['model_sha256']==sha(source/'model.txt')
    z=np.load(source/f'{name}_features.npz');assert np.array_equal(z['id'],df.id) and 'orig_teacher_lgb' in z['columns'];teacher=z['values'][:,list(z['columns']).index('orig_teacher_lgb')];assert ((teacher>=0)&(teacher<=1)).all()
    out=pd.concat([raw,aux],axis=1);out['safe_original_teacher_probability']=teacher.astype('float32');assert out.shape==(len(df),49) and 'id' not in out and 'satisfaction' not in out
    return df.id.to_numpy(),out

class ContextFeatures:
    def fit(self,frame):
        self.columns=list(frame);self.cats=frame.select_dtypes(exclude='number').columns.tolist();assert self.cats==['Gender','Customer Type','Type of Travel','Class']
        self.categories={c:sorted(frame[c].fillna('__MISSING__').astype(str).unique()) for c in self.cats};self.categorical_indices=[self.columns.index(c) for c in self.cats];self.n=len(frame);return self
    def transform(self,frame):
        assert list(frame)==self.columns;out=frame.copy()
        for c in self.cats:out[c]=pd.Categorical(frame[c].fillna('__MISSING__').astype(str),categories=self.categories[c]).codes.astype('float32')
        values=out.to_numpy('float32');assert values.shape==(len(frame),49) and not np.isinf(values).any();return values # native TabPFN handles numeric NaNs
