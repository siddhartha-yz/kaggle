"""Fresh reconstruction of one fixed context; match saved predictions before test."""
from research_r36_worker import *
import copy
from tabicl._model import layers

def operator_proof(model):
    # One largest official FFN block alone avoids retaining a fitted GPU model.
    blocks=[b for b in model.model_.modules() if isinstance(b,layers.MultiheadAttentionBlock)]
    block=copy.deepcopy(max(blocks,key=lambda b:b.linear1.in_features)).to('cpu').eval()
    return block

def main(stage,k,j):
    assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF')=='expandable_segments:True'
    final=json.loads((ROOT/'research_r36_final_inference_plan.json').read_text());plan=json.loads((ROOT/'research_r36_plan.json').read_text());assert final['main_plan_sha256']==sha(ROOT/'research_r36_plan.json')
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**final['code_sha256'],**final['inherited_artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    assert json.loads((ROOT/'research_r36_metadata_audit.json').read_text())['verified_context_records']==32
    run=ROOT/f'research_artifacts/r36_tabicl48_{stage}/fold{k}/context{j}';rec=json.loads((run/'result.json').read_text());ctx=np.load(run/'context.npz')['idx'];v=np.load(run/'validation.npz');out=ROOT/'research_artifacts/r36_replays';out.mkdir(exist_ok=True);destination=out/f'{stage}_fold{k}_context{j}.json'
    if destination.exists():
        previous=json.loads(destination.read_text());assert previous['status']=='passed' and previous['final_plan_sha256']==sha(ROOT/'research_r36_final_inference_plan.json');return
    peak=[0];stop=threading.Event();watcher=threading.Thread(target=watch,args=(stop,peak,out/f'{stage}_{k}_{j}_resource_failure.json'),daemon=True);watcher.start()
    try:
        torch.set_num_threads(6);torch.set_num_interop_threads(1);assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+1024**3;torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        df=pd.read_csv(ROOT/'data/train.csv');raw=df.drop(columns=['id','satisfaction']);y=df.satisfaction.to_numpy('int8');X=features(raw,auxiliary(df.id));cfg=dict(config(ROOT,n_estimators=1,cache=True),norm_methods=[rec['norm_method']],random_state=42+j,offload_mode='cpu',inference_config=inference_config())
        with threadpool_limits(6),streamed_ffn():
            m=FrozenTabICL(**cfg);m.fit(X.iloc[ctx],y[ctx]);assert np.array_equal(m.classes_,[0,1]);query=v['idx'][:512];p=m.predict_proba(X.iloc[query])[:,1];err=float(np.max(np.abs(p-v['pred'][:512])));assert err<1e-6
            block=operator_proof(m) if (stage,k,j)==('fivefold',0,0) else None
        del m,X;gc.collect();torch.cuda.empty_cache();op_error=None
        if block is not None:
            # Same pointwise4096 kernels, sameFP16; large32768-outputcopy equality.
            block.to('cuda:0');g=torch.Generator(device='cpu').manual_seed(20261031);x=torch.randn(1,32768,block.linear1.in_features,generator=g).to('cuda:0');official=layers.MultiheadAttentionBlock._ff_block
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                reference=torch.cat([official(block,piece) for piece in x.split(4096,dim=-2)],dim=-2)
                with streamed_ffn():actual=block._ff_block(x)
                op_error=float(torch.max(torch.abs(reference-actual)).cpu());assert torch.equal(reference,actual) and op_error==0
            del block,x,reference,actual;gc.collect();torch.cuda.empty_cache()
        r=dict(status='passed',completed_utc=now(),stage=stage,fold=k,context=j,context_sha256=sha(run/'context.npz'),prediction_sha256=sha(run/'validation.npz'),reconstructed_512_probability_error=err,large32768_pointwise_FFN_bitwise_error=op_error,peak_gpu_process_bytes=peak[0],final_plan_sha256=sha(ROOT/'research_r36_final_inference_plan.json'),positive_class='True=1');assert 0<peak[0]<=1073741824;dump(destination,r);print(r,flush=True)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
