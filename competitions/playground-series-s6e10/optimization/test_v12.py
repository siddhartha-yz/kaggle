"""Failure boundaries for the supported read-only release verifier and exclusive CSV replay."""
import csv,hashlib,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('v12',Path(__file__).with_name('v12.py'));v12=importlib.util.module_from_spec(spec);spec.loader.exec_module(v12)
class BoundaryTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
 def csv(self,name,rows):
  p=self.root/name
  with p.open('w',newline='') as f:
   w=csv.writer(f,lineterminator='\n');w.writerow(['id','satisfaction']);w.writerows(rows)
  return p
 def test_rejects_corruption_and_missing_file(self):
  p=self.root/'artifact';p.write_bytes(b'good');m={'required_files':{'artifact':v12.sha(p)}};v12.check_hashes(self.root,m);p.write_bytes(b'bad')
  with self.assertRaisesRegex(v12.VerificationError,'SHA256'):v12.check_hashes(self.root,m)
  p.unlink()
  with self.assertRaisesRegex(v12.VerificationError,'Missing'):v12.check_hashes(self.root,m)
 def test_rejects_path_escape(self):
  with self.assertRaisesRegex(v12.VerificationError,'escapes'):v12.inside(self.root,'../outside')
 def test_rejects_csv_id_order(self):
  a=self.csv('a.csv',[(2,.2),(1,.1)]);b=self.csv('b.csv',[(1,0),(2,0)])
  with self.assertRaisesRegex(v12.VerificationError,'ID/order'):v12.inspect_csv(a,b,2)
 def test_rejects_bad_probability(self):
  b=self.csv('sample.csv',[(1,0)])
  for value in ['nan','inf',-1,1.1]:
   with self.subTest(value=value):
    with self.assertRaisesRegex(v12.VerificationError,'probability'):v12.inspect_csv(self.csv('bad.csv',[(1,value)]),b,1)
 def test_rejects_failed_dual_gate(self):
  for name,data in [('audit',{'status':'passed','verified_model_records':8,'verified_native_neural_members':256}),('delivery',{'status':'passed','sha256':'abc','rows':1}),('fold',{'eligible':True,'vs_formal_v11':{'pooled':{'conditional_95pct_ci':[-.01,.1],'delta_auc':.01},'corrected_resampled_t':{'lower_bound':.01}}})]:
   (self.root/name).write_text(json.dumps(data))
  m={'csv_sha256':'abc','rows':1,'proofs':{'model_audit':'audit','delivery':'delivery','fivefold':'fold','threefold':'fold'}}
  with self.assertRaisesRegex(v12.VerificationError,'gate'):v12.check_proofs(self.root,m)
 def test_exclusive_replay_and_identical_reuse(self):
  expected=b'id,satisfaction\n1,0.5\n';m={'csv_sha256':hashlib.sha256(expected).hexdigest()};out=self.root/'out.csv'
  with patch.object(v12,'verify',return_value=({'status':'passed'},[1],[.5])):
   self.assertFalse(v12.reproduce(self.root,m,out)['already_identical']);self.assertEqual(out.read_bytes(),expected);self.assertTrue(v12.reproduce(self.root,m,out)['already_identical']);out.write_bytes(b'user file')
   with self.assertRaisesRegex(v12.VerificationError,'overwrite'):v12.reproduce(self.root,m,out)
   self.assertEqual(out.read_bytes(),b'user file')
 def test_bad_serialization_does_not_publish(self):
  out=self.root/'out.csv'
  with patch.object(v12,'verify',return_value=({},[1],[.5])):
   with self.assertRaisesRegex(v12.VerificationError,'bytes differ'):v12.reproduce(self.root,{'csv_sha256':'wrong'},out)
  self.assertFalse(out.exists());self.assertEqual(list(self.root.glob('.v12-*')),[])

try:
 import numpy as np
 from scipy.special import expit,logit
except ImportError:np=None
@unittest.skipIf(np is None,'NumPy/SciPy are required for the component fixture')
class ComponentTests(unittest.TestCase):
 setUp=BoundaryTests.setUp
 def test_component_algebra_and_independent_mismatch_detection(self):
  ids=np.array([10,11,12]);tree=np.array([.2,.5,.8]);ce=np.array([.3,.6,.7]);base=expit(.5*(logit(tree)+logit(ce)));raw=np.array([.1,.55,.9]);bias=.15;cal=expit(logit(raw)+bias);candidate=expit(.875*logit(base)+.125*logit(cal))
  np.savez(self.root/'tree.npz',id=ids,newtree=tree);np.savez(self.root/'ce.npz',id=ids,newNN=ce);np.savez(self.root/'final.npz',id=ids,v11=base,new_calibrated_NN=cal,candidate=candidate)
  groups=[]
  for k in range(5):
   np.savez(self.root/f'g{k}.npz',id=ids,raw_pred=raw,pred=cal);(self.root/f'c{k}.json').write_text(json.dumps({'bias':bias}));groups.append({'prediction':f'g{k}.npz','calibration':f'c{k}.json'})
  m={'recipe':{'clip':1e-6,'v11_logit_weight':.875,'rank_logit_weight':.125},'tree_components':'tree.npz','CE_components':'ce.npz','final_components':'final.npz','groups':groups}
  before=set(sys.modules);v12.components(self.root,m,ids,base,candidate);self.assertNotIn('torch',set(sys.modules)-before)
  np.savez(self.root/'final.npz',id=ids,v11=base,new_calibrated_NN=cal,candidate=candidate+.01)
  with self.assertRaisesRegex(v12.VerificationError,'formula differs'):v12.components(self.root,m,ids,base,candidate)
if __name__=='__main__':unittest.main()
