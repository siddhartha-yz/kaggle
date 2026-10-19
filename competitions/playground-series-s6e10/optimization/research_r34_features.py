"""Raw official21 covariates plus audited27 target-free conditional rating fields."""
def features(raw,aux):
    assert raw.shape[1]==21 and aux.shape==(len(raw),27) and 'id' not in raw and 'satisfaction' not in raw
    out=raw.reset_index(drop=True).copy()
    for c in aux:assert c not in out;out[c]=aux[c].to_numpy('float32')
    assert out.shape[1]==48
    return out
