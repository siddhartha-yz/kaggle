"""Lean47 plus one independently original-trained LGB teacher logit."""
import numpy as np
from research_r4_lean_features import LeanFeatures

class TeacherFeatures(LeanFeatures):
    @staticmethod
    def append_teacher(out,teacher):
        q=np.asarray(teacher,dtype='float64')
        assert q.ndim==1 and len(q)==len(out) and np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
        q=np.clip(q,1e-6,1-1e-6)
        out['original_teacher_logit']=np.log(q/(1-q)).astype('float32')
        assert out.shape[1]==48 and out.select_dtypes(include='number').shape[1]==26
        return out
    def fit_transform(self,raw,y,teacher):
        return self.append_teacher(super().fit_transform(raw,y),teacher)
    def transform(self,raw,teacher):
        return self.append_teacher(super().transform(raw),teacher)
