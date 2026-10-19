"""TabM16 with CPU-backed data, fixed FP32 batches, bounded GPU and persisted curves."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['MPLCONFIGDIR']='/tmp/kaggle-s6e10-mpl'
import copy,time,gc,threading,resource
import numpy as np,torch
from sklearn.metrics import roc_auc_score
from pytabkit.models.nn_models.tabm import Model,make_parameter_groups
from pytabkit.models.nn_models.rtdl_num_embeddings import compute_bins
from research_w3_common import ROOT,now,dump,sha
from research_r23_guard import watch
CAP=778043392

def setup(run):
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+int(1.25*1024**3)
    torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    peak=[0];stop=threading.Event();watcher=threading.Thread(target=watch,args=(stop,peak,run/'resource_failure.json'),daemon=True);watcher.start()
    return peak,stop,watcher

def predict(model,arrays,batch=512,device='cuda:0'):
    model.eval();out=[]
    with torch.inference_mode():
        for j in range(0,len(arrays[0]),batch):
            n=torch.from_numpy(arrays[0][j:j+batch]).to(device);c=torch.from_numpy(arrays[1][j:j+batch]).to(device)
            out.append(model(n,c).softmax(-1).mean(1)[:,1].cpu().numpy())
    return np.concatenate(out)

def load(path,device='cuda:0'):
    z=torch.load(path,map_location='cpu',weights_only=True);m=Model(**z['args']);m.load_state_dict(z['state_dict']);return m.to(device)

def fit(X,y,cfg,cardinalities,run,name,valid=None,selection=None):
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    n=torch.from_numpy(X[0]);c=torch.from_numpy(X[1]);target=torch.as_tensor(y,dtype=torch.long)
    assert n.device.type==c.device.type==target.device.type=='cpu'
    bins=compute_bins(n,n_bins=cfg['num_bins'])
    args=dict(n_num_features=n.shape[1],cat_cardinalities=cardinalities,n_classes=2,backbone=dict(type='MLP',n_blocks=cfg['n_blocks'],d_block=cfg['d_block'],dropout=cfg['dropout']),bins=bins,num_embeddings=dict(type='PiecewiseLinearEmbeddings',d_embedding=cfg['d_embedding'],activation=False,version='B'),arch_type='tabm',k=cfg['k'],share_training_batches=True)
    m=Model(**args).to('cuda:0');opt=torch.optim.AdamW(make_parameter_groups(m),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    rec=dict(status='training',config=cfg,backing_device='cpu',parameters_device='cuda:0',bins_device='cpu_before_model_transfer',curve=[],started_utc=now())
    best=-np.inf;best_epoch=0;bad=0;tick=time.monotonic()
    for epoch in range(1,cfg['epochs']+1):
        t=time.monotonic();m.train();loss_sum=0.;count=0
        # Preserve the original TabM GPU randperm generator; transfer only indices for CPU slicing.
        for ix_gpu in torch.randperm(len(y),device='cuda:0').split(cfg['batch_size']):
            ix=ix_gpu.cpu();bn=n[ix].to('cuda:0');bc=c[ix].to('cuda:0');by=target[ix].to('cuda:0')
            opt.zero_grad(set_to_none=True);logits=m(bn,bc)
            loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),by.repeat_interleave(cfg['k']))
            assert torch.isfinite(loss);loss.backward();opt.step();loss_sum+=float(loss.detach())*len(ix);count+=len(ix)
        torch.cuda.synchronize();row=dict(epoch=epoch,seconds=time.monotonic()-t,train_loss=loss_sum/count)
        if valid is not None:
            p=predict(m,valid[0]);a=float(roc_auc_score(valid[1],p));row['inner_validation_auc']=a
            if a>best:best=a;best_epoch=epoch;bad=0
            else:bad+=1
            rec.update(best_epoch=best_epoch,best_inner_auc=best,consecutive_nonimproving=bad)
        assert torch.cuda.mem_get_info()[0]>=1024**3
        rec['curve'].append(row);dump(run/f'{name}.json',rec);print(name,row,flush=True)
        if selection is not None and bad>=selection['patience']:break
    rec.update(status='complete',completed_utc=now(),seconds=time.monotonic()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    savedargs=copy.deepcopy(args);savedargs['bins']=[b.cpu() for b in args['bins']]
    torch.save(dict(args=savedargs,state_dict={k:v.detach().cpu() for k,v in m.state_dict().items()}),run/f'{name}.pt')
    rec['model_sha256']=sha(run/f'{name}.pt');dump(run/f'{name}.json',rec)
    del opt;gc.collect();return m,rec
