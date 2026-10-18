"""One frozen source-exemplar contrast; no per-fold quality scoring or tuning."""
from research_r4_common import ROOT, now, dump, sha, guard, evidence
import os, json, time, gc, resource, signal, hashlib
import joblib, numpy as np, lightgbm as lgb
from threadpoolctl import threadpool_limits
from research_r5_auto_te import data
from research_r9_model_features import LinkageFeatures
from research_original import safe_original
from research_external_ablation import external
from research_oof import params

WEIGHT = 30/128

def fit(stage, source, labels, plan):
    guard()
    df, raw, y, fold, official, old_lgb = data(stage)
    oldpath = 'r4_lean_gpu_init_oof' if stage == 'fivefold' else 'r4_lean_gpu_init_replication3'
    oldnn = np.load(ROOT/'research_artifacts'/oldpath/'oof.npz')['pred']
    reference = .5*(official-.25*oldnn)/.75+.5*oldnn
    ext, _ = external('both',df.id)
    out = ROOT/'research_artifacts'/f'r9_linkage_{stage}'
    out.mkdir(exist_ok=True)
    predall = np.full(len(y),np.nan)
    for k in sorted(np.unique(fold)):
        k = int(k)
        fi, vi = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        run = out/f'fold{k}'
        run.mkdir(exist_ok=True)
        record = run/'result.json'
        if record.exists() and json.loads(record.read_text()).get('status') == 'complete':
            rec = json.loads(record.read_text())
            assert rec['runner_sha256'] == sha(__file__) and rec['model_sha256'] == sha(run/'model.txt')
            z = np.load(run/'validation.npz')
            assert np.array_equal(z['id'],df.id.iloc[vi]) and np.array_equal(z['target'],y[vi])
            predall[vi] = z['pred']
            continue
        guard()
        def timeout(signum,frame): raise TimeoutError('Frozen R9 per-fit resource timeout')
        signal.signal(signal.SIGALRM,timeout)
        signal.alarm(plan['per_fit_timeout_seconds'])
        tick = time.monotonic()
        rec = dict(status='training',started_utc=now(),stage=stage,fold=k,
            train_rows=len(fi),valid_rows=len(vi),params=params('stats639'),
            train_index_sha256=hashlib.sha256(fi.tobytes()).hexdigest(),
            runner_sha256=sha(__file__),plan_sha256=sha(ROOT/'research_r9_plan.json'),
            validation_covariates_and_targets_excluded_from_all_fits=True)
        dump(record,rec)
        dump(ROOT/'research_r9_state.json',dict(rec,pid=os.getpid(),active_training=True))
        print('START',stage,k,flush=True)
        try:
            with threadpool_limits(6):
                tr = LinkageFeatures(source,labels)
                X = tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],ext.iloc[fi].reset_index(drop=True))
                dump(run/'linkage_fit.json',dict(summary=tr.linkage.fit_summary,trace=tr.linkage.trace))
                print('PREPROCESSED_TRAIN',stage,k,time.monotonic()-tick,flush=True)
                V = tr.transform(raw.iloc[vi].reset_index(drop=True),ext.iloc[vi].reset_index(drop=True))
                joblib.dump(tr,run/'transformer.joblib')
                model = lgb.LGBMClassifier(**params('stats639'))
                model.fit(X,y[fi])
                pred = model.predict_proba(V)[:,1]
                model.booster_.save_model(str(run/'model.txt'))
                loaded = lgb.Booster(model_file=str(run/'model.txt'))
                error = float(np.max(np.abs(pred-loaded.predict(V,num_threads=6))))
                assert error < 1e-12 and np.isfinite(pred).all()
            predall[vi] = pred
            candidate = reference[vi]+WEIGHT*(pred-old_lgb[vi])
            assert ((candidate>=0)&(candidate<=1)).all()
            np.savez_compressed(run/'validation.npz',id=df.id.iloc[vi].to_numpy(),target=y[vi],
                                idx=vi,pred=pred,candidate=candidate)
            rec.update(status='complete',ended_utc=now(),seconds=time.monotonic()-tick,
                feature_width=64,model_sha256=sha(run/'model.txt'),
                transformer_sha256=sha(run/'transformer.joblib'),prediction_sha256=sha(run/'validation.npz'),
                model_reload_error=error,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            assert rec['peak_rss_bytes'] < plan['peak_rss_ceiling_bytes']
            dump(record,rec)
            print('DONE',stage,k,rec['seconds'],flush=True)
            del tr, X, V, model, loaded
            gc.collect()
        except BaseException as error:
            rec.update(status='failed',error=repr(error),ended_utc=now())
            dump(record,rec)
            raise
        finally:
            signal.alarm(0)
    assert np.isfinite(predall).all()
    candidate = reference+WEIGHT*(predall-old_lgb)
    np.savez_compressed(out/'oof.npz',id=df.id.to_numpy(),target=y,fold=fold,pred=predall,
        candidate=candidate,reference_half=reference,reference_v7=official,old_lgb=old_lgb)
    control = evidence(y,reference,candidate,fold)
    direct = evidence(y,official,candidate,fold)
    result = dict(status='complete',stage=stage,ended_utc=now(),
        vs_private_half=control,vs_official_v7=direct,oof_sha256=sha(out/'oof.npz'),
        eligible=control['eligible_for_new_partition_replication'] and direct['eligible_for_new_partition_replication'])
    dump(out/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result

def main():
    plan = json.loads((ROOT/'research_r9_plan.json').read_text())
    for name,digest in plan['source_sha256'].items(): assert sha(ROOT/name) == digest
    assert json.loads((ROOT/'research_r9_probe.json').read_text())['status'] == 'passed_no_quality_score'
    original, _, _, cols, sourceaudit = safe_original()
    source, labels = original[cols], original.satisfaction.to_numpy('int8')
    assert sourceaudit['safe_rows'] == 129859 and sourceaudit['excluded_exact_rows'] == 21
    result = fit('fivefold',source,labels,plan)
    if result['eligible']: result = fit('threefold',source,labels,plan)
    final = dict(updated_utc=now(),status='complete',last_stage=result['stage'],
        eligible_for_candidate=result['stage']=='threefold' and result['eligible'],
        active_training=False,actual_top100_verified=False,whole_top100_task_completed=False,
        formal_used=1,remaining=1)
    dump(ROOT/'research_r9_result.json',final)
    dump(ROOT/'research_r9_state.json',final)

if __name__ == '__main__':
    try: main()
    except BaseException as error:
        dump(ROOT/'research_r9_failure.json',dict(status='failed',ended_utc=now(),error=repr(error),active_training=False))
        raise
