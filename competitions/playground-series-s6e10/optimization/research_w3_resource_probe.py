"""No-score check of a lower allocator cap; exact frozen TabM architecture and batch."""
from research_w3_common import *
from research_w3_tabm_values import fit,predict
from research_w3_tabm_values_features import TabMValueFeatures
from threadpoolctl import threadpool_limits
def main():
 start_guard();hw=setup_gpu();df,raw,y,splits,_,_,_=load_split('primary');fi=splits[0][0];out=ROOT/'research_artifacts/w3_tabm_resource_probe_1GiB';out.mkdir(exist_ok=False);rec={'status':'running','started_utc':now(),'hardware':hw,'scores_evaluated':False}
 try:
  with threadpool_limits(6):
   tr=TabMValueFeatures();X=tr.fit_transform(raw.iloc[fi]);rec['cat_cardinalities']=tr.cardinalities;m=fit(X,y[fi],plan()['tabm_config'],out,rec,probe=True);pred=predict(m,(X[0][:1024],X[1][:1024]));assert np.isfinite(pred).all()
  rec.update(status='complete',completed_utc=now());dump(out/'result.json',rec)
 except BaseException as e:rec.update(status='failed',error=repr(e),ended_utc=now());dump(out/'result.json',rec);raise
if __name__=='__main__':main()
