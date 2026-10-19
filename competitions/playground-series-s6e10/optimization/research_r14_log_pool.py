"""One frozen equal log-odds opinion pool of existing tree and neural families."""
from research_r4_common import ROOT,now,dump,sha,evidence
import json,numpy as np,pandas as pd
from scipy.special import expit,logit
EPS=1e-6

def pool(tree,neural):
    return expit(.5*logit(np.clip(np.asarray(tree,dtype='float64'),EPS,1-EPS))+
                 .5*logit(np.clip(np.asarray(neural,dtype='float64'),EPS,1-EPS)))

def independent_pool(tree,neural):
    t=np.clip(np.asarray(tree,dtype='float64'),EPS,1-EPS)
    n=np.clip(np.asarray(neural,dtype='float64'),EPS,1-EPS)
    positive=np.sqrt(t*n);negative=np.sqrt((1-t)*(1-n))
    return positive/(positive+negative)

def main():
    protocol=json.loads((ROOT/'research_r14_plan.json').read_text())
    assert protocol['source_sha256']==sha(__file__)
    assert sha(ROOT/'data/train.csv')=='2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36'
    for path,digest in protocol['prediction_sources_sha256'].items():assert sha(ROOT/path)==digest
    train=pd.read_csv(ROOT/'data/train.csv');y=train.satisfaction.to_numpy('int8');stages={}
    for stage,path in [('fivefold','r4_lean_gpu_init_oof'),('threefold','r4_lean_gpu_init_replication3')]:
        z=np.load(ROOT/'research_artifacts'/path/'oof.npz')
        assert np.array_equal(z['id'],train.id) and np.array_equal(z['target'],y)
        if stage=='fivefold':
            from research_nested_stack import outer_predictions
            core=outer_predictions('oof',np.arange(len(y)),train,z['fold'])
            deep=np.load(ROOT/'research_artifacts/cb_deep_oof/raw_oof.npz');assert np.array_equal(deep['id'],train.id)
            core=np.column_stack([core,deep['pred']])
        else:
            base=np.load(ROOT/'research_artifacts/r4_baseline_replication3/oof.npz')
            assert np.array_equal(base['id'],train.id) and np.array_equal(base['fold'],z['fold']);core=base['components'][:,:4]
        tree=core@np.array([.46875,.09375,.1875,.25]);neural=z['pred'];half=.5*tree+.5*neural
        assert np.max(np.abs(.75*tree+.25*neural-z['candidate']))<1e-12
        candidate=pool(tree,neural);error=float(np.max(np.abs(candidate-independent_pool(tree,neural))))
        assert error<1e-12 and np.isfinite(candidate).all() and ((candidate>=0)&(candidate<=1)).all()
        result=dict(stage=stage,checked_utc=now(),vs_private_half=evidence(y,half,candidate,z['fold']),
            vs_official_v7=evidence(y,z['candidate'],candidate,z['fold']),independent_formula_max_error=error)
        result['eligible']=all(result[key]['eligible_for_new_partition_replication'] for key in ('vs_private_half','vs_official_v7'))
        out=ROOT/'research_artifacts'/f'r14_log_pool_{stage}';out.mkdir(exist_ok=True)
        np.savez_compressed(out/'oof.npz',id=train.id.to_numpy(),target=y,fold=z['fold'],tree=tree,neural=neural,candidate=candidate,reference_half=half)
        result['oof_sha256']=sha(out/'oof.npz');dump(out/'result.json',result);stages[stage]=result
        print(stage,result['vs_private_half']['pooled'],result['eligible'],flush=True)
        if not result['eligible']:break
    qualified=len(stages)==2 and all(result['eligible'] for result in stages.values())
    final=dict(status='complete',updated_utc=now(),stages=stages,eligible_for_candidate=qualified,
        new_models_fitted=0,one_fixed_function_no_weight_grid=True,formal_used=1,remaining=1,
        actual_top100_verified=False,whole_top100_task_completed=False,active_training=False)
    dump(ROOT/'research_r14_result.json',final)
    if not qualified:return
    # Only gated cached inference, no fitting, and independent component and CSV reconstruction.
    assert json.loads((ROOT/'research_r4_delivery_audit.json').read_text())['status']=='passed'
    for name,digest in protocol['test_sources_sha256'].items():assert sha(ROOT/name)==digest
    test=pd.read_csv(ROOT/'data/test.csv');sample=pd.read_csv(ROOT/'data/sample_submission.csv')
    core=np.load(ROOT/'research_artifacts/cb_deep_full_v1/test_components.npz')
    old=np.load(ROOT/'research_artifacts/r4_lean_cv_final/test_components.npz')
    assert np.array_equal(core['id'],test.id) and np.array_equal(old['id'],test.id) and np.array_equal(sample.id,test.id)
    tree=(60*core['lgb_original']+12*core['cb_original']+24*core['residual_prior05']+32*core['cb_deep'])/128
    assert np.max(np.abs(tree-old['v5']))<1e-12
    candidate=pool(tree,old['realmlp']);assert np.max(np.abs(candidate-independent_pool(tree,old['realmlp'])))<1e-12
    oldcsv=pd.read_csv(ROOT/'submission_lean_half_r6_v8.csv')
    assert np.array_equal(oldcsv.id,test.id) and np.max(np.abs(oldcsv.satisfaction-(.5*tree+.5*old['realmlp'])))<1e-14
    csv=ROOT/'submission_log_odds_pool_r14_v9.csv'
    if not csv.exists():pd.DataFrame({'id':test.id,'satisfaction':candidate}).to_csv(csv,index=False,float_format='%.17g')
    check=pd.read_csv(csv);assert len(check)==299844 and list(check)==list(sample) and np.array_equal(check.id,test.id)
    error=float(np.max(np.abs(check.satisfaction-candidate)));assert error<1e-14 and check.satisfaction.between(0,1).all()
    delivery=dict(status='passed',audited_utc=now(),csv=csv.name,sha256=sha(csv),rows=len(check),bytes=csv.stat().st_size,
        positive_class=True,family_weights={'v5_tree_logodds':.5,'leanCV40_logodds':.5},clip_epsilon=EPS,
        formula='sigmoid(.5logit(tree)+.5logit(mean40NN)); family pooling follows model averaging',
        inherited_model_audit_sha256=sha(ROOT/'research_r4_delivery_audit.json'),new_models_fitted=0,
        validation=stages,CSV_roundtrip_error=error,plan_sha256=sha(ROOT/'research_r14_plan.json'),
        formal_used=1,remaining=1,parent_only_submitter=True,actual_top100_verified=False,
        limitation='Reused validation labels and external50/50prior; conditional bounds omit adaptive selection. Nonlinear pool after averaging may differ from averaging pools; final order fixed before scoring.')
    dump(ROOT/'research_r14_delivery_audit.json',delivery)
    print(json.dumps({k:v for k,v in delivery.items() if k!='validation'},indent=2),flush=True)
if __name__=='__main__':main()
