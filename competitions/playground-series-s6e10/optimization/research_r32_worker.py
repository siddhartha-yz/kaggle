"""One independent nested TabM74 fit per fresh process, no outer epoch selection."""
from research_r32_engine import *
from research_r32_features import ConditionalTabMFeatures
from research_r23_loader import auxiliary
import json,sys,pandas as pd,joblib,hashlib
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

def main(stage,k):
    plan=json.loads((ROOT/'research_r32_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    if stage=='threefold':assert json.loads((ROOT/'research_artifacts/r32_tabm74_fivefold/result.json').read_text())['eligible']
    df=pd.read_csv(ROOT/'data/train.csv');raw=df.drop(columns=['id','satisfaction']);y=df.satisfaction.to_numpy('int8')
    z=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y)
    fi=np.flatnonzero(z['fold']!=k);vi=np.flatnonzero(z['fold']==k);aux=auxiliary(df.id)
    out=ROOT/f'research_artifacts/r32_tabm74_{stage}';out.mkdir(exist_ok=True);run=out/f'fold{k}';run.mkdir(exist_ok=False)
    peak,stop,watcher=setup(run)
    try:
        with threadpool_limits(6):
            it,im=train_test_split(fi,test_size=plan['selection']['inner_valid_fraction'],stratify=y[fi],random_state=plan['selection']['split_seed'])
            assert not np.intersect1d(it,im).size and not np.intersect1d(np.r_[it,im],vi).size
            np.savez_compressed(run/'inner_split.npz',fit_global=it,valid_global=im)
            inner=ConditionalTabMFeatures();IX=inner.fit_transform(raw.iloc[it].reset_index(drop=True),y[it],aux.iloc[it].reset_index(drop=True));IM=inner.transform(raw.iloc[im].reset_index(drop=True),aux.iloc[im].reset_index(drop=True));joblib.dump(inner,run/'selection_transformer.joblib')
            model,selected=fit(IX,y[it],plan['config'],inner.cardinalities,run,'selection',valid=(IM,y[im]),selection=plan['selection'])
            best=selected['best_epoch'];assert best==int(np.argmax([r['inner_validation_auc'] for r in selected['curve']]))+1
            del model,inner,IX,IM;gc.collect();torch.cuda.empty_cache()
            tr=ConditionalTabMFeatures();X=tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],aux.iloc[fi].reset_index(drop=True));V=tr.transform(raw.iloc[vi].reset_index(drop=True),aux.iloc[vi].reset_index(drop=True));joblib.dump(tr,run/'transformer.joblib')
            rec=dict(status='training',started_utc=now(),stage=stage,fold=k,selected_epochs=best,train_rows=len(fi),valid_rows=len(vi),feature_width=74,numeric_width=X[0].shape[1],cat_cardinalities=tr.cardinalities,outer_validation_used_for_fit_or_selection=False,train_index_sha256=hashlib.sha256(fi.tobytes()).hexdigest(),plan_sha256=sha(ROOT/'research_r32_plan.json'),runner_sha256=sha(__file__))
            dump(run/'result.json',rec);model,fitrec=fit(X,y[fi],dict(plan['config'],epochs=best),tr.cardinalities,run,'model')
            p=predict(model,V);assert np.isfinite(p).all();del model;gc.collect();torch.cuda.empty_cache()
            restored=load(run/'model.pt');again=predict(restored,(V[0][:2048],V[1][:2048]));err=float(np.max(np.abs(p[:2048]-again)));assert err<1e-7
            np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=p)
            assert 0<peak[0]<=1073741824
            rec.update(status='complete',completed_utc=now(),model_sha256=sha(run/'model.pt'),transformer_sha256=sha(run/'transformer.joblib'),prediction_sha256=sha(run/'validation.npz'),selection_sha256=sha(run/'selection.json'),selection_model_sha256=sha(run/'selection.pt'),selection_transformer_sha256=sha(run/'selection_transformer.joblib'),inner_split_sha256=sha(run/'inner_split.npz'),peak_gpu_process_bytes=peak[0],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,model_reload_error=err,fit_seconds=fitrec['seconds'])
            dump(run/'result.json',rec);print('FOLD_COMPLETE',stage,k,rec,flush=True)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))
