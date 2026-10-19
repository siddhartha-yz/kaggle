"""Supported v12 artifact verification and exact CSV replay; no training, pickle loads or network."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,tempfile
from itertools import zip_longest
from pathlib import Path

HERE=Path(__file__).resolve().parent
class VerificationError(ValueError):pass

def require(condition,message):
    if not condition:raise VerificationError(message)
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def inside(root,name):
    path=(root/name).resolve()
    require(path.is_relative_to(root),'Artifact path escapes root: '+name)
    return path

def check_hashes(root,manifest):
    for name,expected in manifest['required_files'].items():
        path=inside(root,name);require(path.is_file(),'Missing artifact: '+name);require(sha(path)==expected,'SHA256 mismatch: '+name)

def inspect_csv(path,sample_path,rows):
    identifiers=[];values=[]
    with path.open(newline='') as f,sample_path.open(newline='') as sf:
        data=csv.DictReader(f);sample=csv.DictReader(sf)
        require(data.fieldnames==sample.fieldnames==['id','satisfaction'],'Unexpected CSV schema')
        for number,(row,reference) in enumerate(zip_longest(data,sample),1):
            require(row is not None and reference is not None,'CSV/sample row count differs')
            require(set(row)=={'id','satisfaction'} and row['id']==reference['id'],'ID/order or row schema mismatch at row '+str(number))
            try:value=float(row['satisfaction']);identifier=int(row['id'])
            except (TypeError,ValueError) as e:raise VerificationError('Invalid scalar at row '+str(number)) from e
            require(math.isfinite(value) and 0<=value<=1,'Invalid probability at row '+str(number));identifiers.append(identifier);values.append(value)
    require(len(values)==rows and len(set(identifiers))==rows,'Wrong count or duplicate IDs')
    return identifiers,values

def check_proofs(root,manifest):
    p=manifest['proofs'];audit=json.loads(inside(root,p['model_audit']).read_text());delivery=json.loads(inside(root,p['delivery']).read_text())
    require(audit['status']=='passed' and audit['verified_model_records']==8 and audit['verified_native_neural_members']==256,'Eight-model audit is not passed')
    require(delivery['status']=='passed' and delivery['sha256']==manifest['csv_sha256'] and delivery['rows']==manifest['rows'],'Delivery proof differs')
    scores={}
    for stage in ['fivefold','threefold']:
        record=json.loads(inside(root,p[stage]).read_text());proof=record['vs_formal_v11'];paired=proof['pooled']['conditional_95pct_ci'][0];corrected=proof['corrected_resampled_t']['lower_bound']
        require(record['eligible'] and paired>0 and corrected>0,stage+' validation gate failed');scores[stage]={'delta_auc':proof['pooled']['delta_auc'],'paired_lower':paired,'corrected_lower':corrected}
    return scores

def components(root,manifest,identifiers,baseline,actual):
    # These imports stay out of describe/help and use one CPU thread. No model class is imported.
    for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
    import numpy as np
    from scipy.special import expit,logit
    ids=np.asarray(identifiers,dtype=np.int64);baseline=np.asarray(baseline);actual=np.asarray(actual);eps=manifest['recipe']['clip'];require(eps==1e-6 and manifest['recipe']['v11_logit_weight']==.875 and manifest['recipe']['rank_logit_weight']==.125,'Unsupported v12 formula')
    def read(name):
        with np.load(inside(root,name),allow_pickle=False) as z:return {k:z[k] for k in z.files}
    def check_ids(z):require(np.array_equal(z['id'],ids),'Component ID/order differs')
    tree=read(manifest['tree_components']);ce=read(manifest['CE_components']);final=read(manifest['final_components'])
    for z in [tree,ce,final]:check_ids(z)
    pool=lambda a,b:expit(.5*(logit(np.clip(a,eps,1-eps))+logit(np.clip(b,eps,1-eps))))
    baseline_error=float(np.max(np.abs(pool(tree['newtree'],ce['newNN'])-baseline)));require(baseline_error<1e-12,'v11 tree/CE formula differs');groups=[];calibration_error=0.
    require(len(manifest['groups'])==5,'Expected five native32 groups')
    for item in manifest['groups']:
        z=read(item['prediction']);check_ids(z);bias=json.loads(inside(root,item['calibration']).read_text())['bias'];pred=expit(logit(np.clip(z['raw_pred'].astype('float64'),eps,1-eps))+bias);err=float(np.max(np.abs(pred-z['pred'])));require(err<1e-12,'Training-origin calibration differs');calibration_error=max(calibration_error,err);groups.append(z['pred'])
    mean=np.mean(np.asarray(groups),axis=0);mean_error=float(np.max(np.abs(mean-final['new_calibrated_NN'])));require(mean_error<1e-14,'Native-group mean differs');require(np.max(np.abs(final['v11']-baseline))<1e-14,'v11 reference differs')
    a=np.clip(baseline,eps,1-eps);b=np.clip(mean,eps,1-eps);numerator=a**.875*b**.125;independent=numerator/(numerator+(1-a)**.875*(1-b)**.125);formula_error=float(np.max(np.abs(independent-final['candidate'])));CSV_error=float(np.max(np.abs(actual-final['candidate'])));require(formula_error<1e-12 and CSV_error<1e-14,'Final probability or CSV formula differs')
    return {'baseline_formula_error':baseline_error,'calibration_error':calibration_error,'group_mean_error':mean_error,'independent_formula_error':formula_error,'CSV_numeric_error':CSV_error},ids,final['candidate']

def verify(root,manifest):
    root=Path(root).resolve();require(manifest['schema_version']==1 and manifest['positive_class']=='True=1','Unsupported release schema/class')
    check_hashes(root,manifest);scores=check_proofs(root,manifest);path=inside(root,manifest['csv']);require(sha(path)==manifest['csv_sha256'] and path.stat().st_size==manifest['csv_bytes'],'Canonical CSV bytes differ')
    identifiers,actual=inspect_csv(path,inside(root,manifest['sample']),manifest['rows']);ids2,baseline=inspect_csv(inside(root,manifest['baseline_csv']),inside(root,manifest['sample']),manifest['rows']);require(ids2==identifiers,'Baseline IDs differ');errors,ids,pred=components(root,manifest,identifiers,baseline,actual)
    report={'status':'passed','release':manifest['release'],'rows':manifest['rows'],'sha256':manifest['csv_sha256'],'bytes':manifest['csv_bytes'],'verified_files':len(manifest['required_files']),'verification':'Read-only artifact hashes, stored dual-gate/audit proofs, calibration, component algebra and CSV; no training, GPU or model deserialization','validation':scores,'errors':errors,'official_snapshot':manifest['official'],'actual_top500_verified':False}
    return report,ids,pred

def reproduce(root,manifest,output):
    report,ids,pred=verify(root,manifest);output=Path(output).resolve()
    if output.exists():
        require(output.is_file() and sha(output)==manifest['csv_sha256'],'Refusing to overwrite existing output');return dict(report,replayed_csv=str(output),already_identical=True)
    output.parent.mkdir(parents=True,exist_ok=True);temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',newline='',encoding='utf-8',prefix='.v12-',suffix='.csv',dir=output.parent,delete=False) as f:
            temporary=Path(f.name);writer=csv.writer(f,lineterminator='\n');writer.writerow(['id','satisfaction'])
            for identifier,value in zip(ids,pred):writer.writerow([int(identifier),'%.17g'%float(value)])
        require(sha(temporary)==manifest['csv_sha256'],'Replayed CSV bytes differ; original artifacts were not changed')
        os.link(temporary,output)  # Exclusive creation: a concurrently-created user file is never replaced.
        return dict(report,replayed_csv=str(output),already_identical=False)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,default=HERE/'v12_manifest.json');sub=parser.add_subparsers(dest='command',required=True);sub.add_parser('describe',help='Describe frozen recipe/evidence without dependencies or data')
    for name in ['verify','reproduce-csv']:
        p=sub.add_parser(name);p.add_argument('--artifact-root',type=Path,required=True)
        if name=='reproduce-csv':p.add_argument('--output',type=Path,required=True)
    a=parser.parse_args(argv)
    try:
        manifest=json.loads(a.manifest.read_text())
        if a.command=='describe':result={k:manifest[k] for k in ['release','source_boundary','rows','csv_sha256','recipe','official','limitations']}
        elif a.command=='verify':result=verify(a.artifact_root,manifest)[0]
        else:result=reproduce(a.artifact_root,manifest,a.output)
        print(json.dumps(result,indent=2));return 0
    except (VerificationError,OSError,KeyError,ValueError,ImportError) as error:
        print(json.dumps({'status':'failed','error':str(error)}));return 1
if __name__=='__main__':raise SystemExit(main())
