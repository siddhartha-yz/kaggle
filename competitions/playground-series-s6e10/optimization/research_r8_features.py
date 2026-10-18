"""Preserve raw numeric lean view and append cross-fitted conditional responses."""
import numpy as np
from sklearn.preprocessing import TargetEncoder
from research_r7_response_features import ResponseFeatures
from research_r4_lean_features import LeanFeatures


class AppendedResponseFeatures(ResponseFeatures):
    def fit_transform(self, raw, y):
        out = LeanFeatures.fit_transform(self, raw, y)
        self.response_encoder = TargetEncoder(target_type='binary', smooth='auto',
                                             cv=5, shuffle=True, random_state=20261025)
        response = self.response_encoder.fit_transform(self.response_keys(raw), np.asarray(y))
        return self._append(out, response)

    def transform(self, raw):
        out = LeanFeatures.transform(self, raw)
        return self._append(out, self.response_encoder.transform(self.response_keys(raw)))

    @staticmethod
    def _append(out, response):
        assert response.shape == (len(out), 13) and np.isfinite(response).all()
        for j in range(13):
            out[f'conditional_response_{j:02}'] = response[:, j].astype('float32')
        assert out.shape[1] == 60 and out.select_dtypes(include='number').shape[1] == 38
        return out
