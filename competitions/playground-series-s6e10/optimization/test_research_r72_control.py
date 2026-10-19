"""Synthetic checks: only explicit resource failures permit unchanged checkpoint recovery."""
import importlib.util,pathlib,tempfile,subprocess,json,hashlib
root=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r72_control_test_import',root/'research_r72_followthrough.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
cases=0
with tempfile.TemporaryDirectory(prefix='r72-control-test-') as folder:
    module.ROOT=pathlib.Path(folder)
    for kind,script,args,name in [('validation','research_r72_worker.py',['fivefold','0','2'],'research_r72_fivefold_fold0_context2.log'),('audit','research_r72_audit_worker.py',['threefold','2','3'],'research_r72_audit_threefold_fold2_context3.log'),('test','research_r72_test_worker.py',['4','1'],'research_r72_test_fold4_context1.log')]:
        path=module.ROOT/name;cmd=['/fake/python','-u',script,*args]
        path.write_text('');guard=module.resource_failure(repr(subprocess.CalledProcessError(86,cmd)),kind);assert guard['reason']=='taskresourceguard' and guard['args']==args;cases+=1
        path.write_text('assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+MIN_GPU_FREE and host_available()>=MIN_AVAILABLE+2*1024**3\nAssertionError\n');headroom=module.resource_failure(repr(subprocess.CalledProcessError(1,cmd)),kind);assert headroom['reason']=='initialheadroomcheck' and headroom['args']==args;cases+=1
        path.write_text('assert sha(file)==expected\nAssertionError\n');assert module.resource_failure(repr(subprocess.CalledProcessError(1,cmd)),kind) is None;cases+=1
        path.write_text('Unrelated numerical or code failure\nValueError\n');assert module.resource_failure(repr(subprocess.CalledProcessError(1,cmd)),kind) is None;cases+=1
    assert module.resource_failure('unrecognized failure','validation') is None;cases+=1
record={'status':'passed_synthetic_only','cases':cases,'main_worker_sha256':hashlib.sha256((root/'research_r72_worker.py').read_bytes()).hexdigest(),'control_source_sha256':hashlib.sha256((root/'research_r72_followthrough.py').read_bytes()).hexdigest(),'official_quality_used':False,'GPU_work_started':False};(root/'research_r72_control_test.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
