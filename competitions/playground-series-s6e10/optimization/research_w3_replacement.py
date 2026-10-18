"""Bind the exact original LGB component in v6 before any component replacement."""
from research_w3_common import *
from research_nested_stack import outer_predictions
W=45/128
def old_component(split,vi,df,fold,old_nn,reference):
 P=outer_predictions(split,vi,df,fold);p=ROOT/'research_artifacts'/('cb_deep_oof/raw_oof.npz' if split=='oof' else f'cb_deep_{split}/fold0/validation.npz');z=np.load(p)
 if split=='oof':assert np.array_equal(z['id'],df.id) and np.array_equal(z['fold'],fold);deep=z['pred'][vi]
 else:assert np.array_equal(z['idx'],vi) and np.array_equal(z['id'],df.id.iloc[vi]);deep=z['pred']
 reconstructed=(45*P[:,0]+9*P[:,1]+18*P[:,2]+24*deep+32*old_nn)/128;delta=float(np.max(np.abs(reconstructed-reference)));assert delta<1e-12;return P[:,0],delta
