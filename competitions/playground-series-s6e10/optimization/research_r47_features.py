"""Use conditional ranks in 13 numeric coordinates; preserve exact raw-rating categories."""
import numpy as np
from research_r4_lean_features import LeanFeatures
RATINGS=['Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling','Checkin service','Cleanliness']
class RankCoordinateFeatures(LeanFeatures):
    def append(self,out,aux):
        assert aux.shape==(len(out),40) and list(aux)==self.aux_columns and np.isfinite(aux.to_numpy()).all()
        for c in aux.columns[:27]:
            assert c not in out;out[c]=aux[c].to_numpy('float32')
        for j,rating in enumerate(RATINGS):
            assert rating in self.numeric and aux.columns[27+j]==f'conditional_positive_rank_{j}'
            identity=f'value_cat_{self.numeric.index(rating)}'
            # Discrete identities still carry all original 0..5 information.
            assert np.array_equal(out[identity].astype('int64').to_numpy(),out[rating].to_numpy())
            out[rating]=aux.iloc[:,27+j].to_numpy('float32')
        assert out.shape[1]==74 and out.select_dtypes(include='number').shape[1]==52
        return out
    def fit_transform(self,raw,y,aux):
        self.aux_columns=list(aux);return self.append(super().fit_transform(raw,y),aux)
    def transform(self,raw,aux):return self.append(super().transform(raw),aux)
