"""Synthetic parser boundary, optimizer-head isolation, schedules and checkpoint selection."""
from research_r94_model import *
from pathlib import Path
import io,json,hashlib

def main():
 torch.set_num_threads(2);torch.manual_seed(17)
 # Invalid outer labels would raise if parsed. skiprows prevents them reaching parser.
 text='id,satisfaction\n0,False\n1,OUTER_SECRET\n2,True\n3,OUTER_SECRET\n'
 keep=np.array([True,False,True,False])
 own=pd.read_csv(io.StringIO(text),usecols=['satisfaction'],skiprows=lambda row:row>0 and not keep[row-1],dtype={'satisfaction':'bool'})
 assert own.satisfaction.tolist()==[False,True]
 frame=pd.DataFrame({c:np.arange(8,dtype=float) for c in NUM})
 for c in CATS:frame[c]=['a','b']*4
 pre=Preprocessor().fit(frame);n,c=pre.transform(frame);model=RTDModel(pre)
 initial={k:v.clone() for k,v in model.state_dict().items()};head=model.classifier.weight.clone()
 optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
 selection=torch.ones(8,21,dtype=torch.bool);donor=torch.zeros(8,21,dtype=torch.long)
 a,b,t=corrupt(n,c,n,c,selection,donor);logits,_=model(a,b)
 torch.nn.functional.binary_cross_entropy_with_logits(logits,t).backward();optimizer.step()
 assert torch.equal(head,model.classifier.weight) and not torch.equal(initial['numeric_weight'],model.numeric_weight)
 model.load_state_dict(initial);assert all(torch.equal(model.state_dict()[k],v) for k,v in initial.items())
 optimizer=torch.optim.AdamW(model.parameters(),lr=.001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20,eta_min=.00001)
 lrs=[]
 for _ in range(20):lrs.append(optimizer.param_groups[0]['lr']);optimizer.step();scheduler.step()
 assert lrs[0]==.001 and abs(optimizer.param_groups[0]['lr']-.00001)<1e-15
 assert min([dict(epoch=1,inner_logloss=.5),dict(epoch=2,inner_logloss=.5)],key=lambda d:d['inner_logloss'])['epoch']==1
 out=dict(status='passed_synthetic_only',outer_label_parser_exclusion=True,RTD_does_not_update_classifier_head=True,
   initial_state_replay_exact=True,cosine_epoch_start_LRs=lrs,first_epoch_wins_inner_ties=True,official_data_loaded=False)
 root=Path(__file__).resolve().parent;out['test_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 (root/'research_r94_protocol_test.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
