"""Select GPU tree count using only a strict inner split of current training data."""
import gc,json,numpy as np,joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from research_r16_cb_features import CatDistanceFeatures
from research_r4_common import dump,sha,now

def select_rounds(raw,y,ext,global_indices,out,parameters,selection):
    assert len(raw)==len(y)==len(ext)==len(global_indices)
    fi,vi=train_test_split(np.arange(len(y)),test_size=selection['inner_valid_fraction'],stratify=y,random_state=selection['split_seed'])
    assert not np.intersect1d(fi,vi).size and len(fi)+len(vi)==len(y)
    np.savez_compressed(out/'inner_split.npz',fit_global=global_indices[fi],valid_global=global_indices[vi])
    tr=CatDistanceFeatures(False)
    X=tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],ext.iloc[fi].reset_index(drop=True))
    V=tr.transform(raw.iloc[vi].reset_index(drop=True),ext.iloc[vi].reset_index(drop=True))
    joblib.dump(tr,out/'selection_transformer.joblib')
    params=dict(parameters,iterations=selection['max_trees'])
    model=CatBoostClassifier(**params)
    model.fit(X,y[fi],cat_features=tr.cat_columns,eval_set=(V,y[vi]),use_best_model=True,early_stopping_rounds=selection['patience'])
    history=model.get_evals_result()['validation']['AUC']
    best=int(model.get_best_iteration())+1
    assert 1<=best<=selection['max_trees'] and model.tree_count_==best
    assert best==int(np.argmax(history))+1
    model.save_model(str(out/'selection_model.cbm'))
    result=dict(status='selected',completed_utc=now(),selected_trees=best,inner_train_rows=len(fi),inner_valid_rows=len(vi),
        inner_split_sha256=sha(out/'inner_split.npz'),selection_model_sha256=sha(out/'selection_model.cbm'),
        selection_transformer_sha256=sha(out/'selection_transformer.joblib'),history=history,
        outer_validation_used=False,parameters=params,selection=selection)
    dump(out/'selection.json',result)
    print('INNER_SELECTED',best,'history_length',len(history),flush=True)
    del model,tr,X,V;gc.collect()
    return best
