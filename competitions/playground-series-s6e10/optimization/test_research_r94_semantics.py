"""Synthetic only: replacement semantics, synchronized missingness, no positional/row shortcut."""
from research_r94_model import *
import json,hashlib,pathlib,time

def main():
    torch.set_num_threads(2);torch.manual_seed(20261022);start=time.monotonic()
    num=torch.arange(68,dtype=torch.float32).reshape(4,17);num[0,0]=float('nan');num[1,0]=float('nan');num[0,1]=-0.;num[1,1]=0.
    cat=torch.tensor([[0,1,2,3],[0,1,2,4],[2,2,2,2],[3,3,3,3]])
    selected=torch.ones(4,21,dtype=torch.bool);donors=torch.tensor([[1]*21,[0]*21,[2]*21,[3]*21])
    a,b,y=corrupt(num,cat,num,cat,selected,donors)
    for i in range(4):
        for j in range(21):
            left=(num if j<17 else cat)[i,j if j<17 else j-17]
            right=(num if j<17 else cat)[donors[i,j],j if j<17 else j-17]
            same=bool(left==right) or (j<17 and bool(torch.isnan(left)&torch.isnan(right)))
            assert y[i,j].item()==float(not same)
    assert y[0,0]==0 and y[0,1]==0 and torch.isnan(a[0,0])
    a0,b0,y0=corrupt(num,cat,num,cat,torch.zeros_like(selected),donors)
    assert torch.allclose(a0,num,equal_nan=True) and torch.equal(b0,cat) and y0.sum()==0
    frame=pd.DataFrame({c:np.arange(4,dtype=float) for c in NUM})
    for c in CATS:frame[c]=['a','b',None,'a']
    p=Preprocessor().fit(frame);n,c=p.transform(frame);m=RTDModel(p).eval()
    with torch.no_grad():
        r,s=m(n,c);permutation=torch.randperm(21);rp,sp=m(n,c,permutation)
        permutation_error=float((r-rp[:,torch.argsort(permutation)]).abs().max())
        cls_permutation_error=float((s-sp).abs().max())
        solo_r,solo_s=m(n[:1],c[:1]);batch_error=float(max((r[:1]-solo_r).abs().max(),(s[:1]-solo_s).abs().max()))
    assert max(permutation_error,cls_permutation_error,batch_error)<2e-6
    m.train();n2,c2,t=corrupt(n,c,n,c,selected,donors);r,_=m(n2,c2)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(r,t);loss.backward()
    assert m.numeric_weight.grad.abs().sum()>0 and m.encoder.layers[0].linear1.weight.grad.abs().sum()>0
    assert m.classifier.weight.grad is None
    result=dict(status='passed_synthetic_only',replacement_cell_checks=84,selected_same_value_is_negative=True,
        missingness_replaced_atomically=True,no_external_side_features=True,
        token_permutation_error=permutation_error,CLS_permutation_error=cls_permutation_error,
        batch_independence_error=batch_error,RTD_encoder_gradient_nonzero=True,
        official_data_loaded=False,outer_quality_scored=False,seconds=time.monotonic()-start)
    root=pathlib.Path(__file__).resolve().parent
    result['source_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['research_r94_model.py',pathlib.Path(__file__).name]}
    (root/'research_r94_semantics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
