"""Anonymous, pinned upstream TabSTAR/E5 preparation; no model execution."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, shutil, urllib.request

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'research_sources/new_foundation_feasibility'
CACHE = ROOT / 'research_artifacts/r79_hf_offline/hub'
REPOS = [
    ('alana89/TabSTAR', '73560fb23d34307f4b6fa849f38f8299a8980a4a',
     'tabstar_metadata.json', ['README.md', 'config.json', 'model.safetensors']),
    ('intfloat/e5-small-v2', 'ffb93f3bd4047442299a41ebb6fa998a38507c52',
     'e5_small_v2_metadata.json', ['README.md', 'config.json', 'model.safetensors',
      'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt']),
]

def state(status, **kw):
    d = dict(status=status, updated_utc=datetime.now(timezone.utc).isoformat(),
             pid=os.getpid(), classifier_fits=0, GPU_used=False, **kw)
    p = ROOT / 'research_r79_download_state.json'
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(d, indent=2)+'\n'); t.replace(p)
    print(json.dumps(d), flush=True)

def verify(path, info):
    if not path.is_file() or path.stat().st_size != info['size']:
        return False
    h = hashlib.sha256() if 'lfs' in info else hashlib.sha1()
    if 'lfs' not in info:
        h.update(f"blob {info['size']}\0".encode())
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest() == (info['lfs']['sha256'] if 'lfs' in info else info['blobId'])

def main():
    assert shutil.disk_usage(ROOT).free > 2*1024**3, 'disk reserve'
    records = []
    for repo, revision, metadata, names in REPOS:
        m = json.loads((SOURCE/metadata).read_text())
        assert m['id'] == repo and m['sha'] == revision and m['gated'] is False
        base = CACHE / ('models--'+repo.replace('/', '--'))
        snapshot = base/'snapshots'/revision
        snapshot.mkdir(parents=True, exist_ok=True)
        entries = {q['rfilename']:q for q in m['siblings']}
        for name in names:
            info = entries[name]
            assert 0 < info['size'] < 200_000_000
            target = snapshot/name
            if not verify(target, info):
                partial = target.with_name(target.name+'.partial')
                request = urllib.request.Request(
                    f'https://huggingface.co/{repo}/resolve/{revision}/{name}',
                    headers={'User-Agent':'kaggle-local-research/1.0'})
                state('anonymous_pinned_download', repository=repo, revision=revision, filename=name)
                with urllib.request.urlopen(request, timeout=60) as response, partial.open('wb') as f:
                    total, last = 0, 0
                    while True:
                        b = response.read(1024*1024)
                        if not b: break
                        total += len(b)
                        assert total <= info['size'], 'upstream size exceeded'
                        f.write(b)
                        if total-last >= 8*1024*1024:
                            last = total
                            state('anonymous_pinned_download', repository=repo, filename=name,
                                  bytes_received=total, expected_bytes=info['size'])
                assert verify(partial, info), 'upstream content digest/size mismatch'
                partial.replace(target)
            records.append(dict(repository=repo, revision=revision, filename=name,
                                bytes=target.stat().st_size,
                                sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        (base/'refs').mkdir(exist_ok=True)
        (base/'refs/main').write_text(revision)
    receipt = dict(status='all_pinned_files_verified_no_model_execution',
                   checked_utc=datetime.now(timezone.utc).isoformat(), files=records,
                   TabSTAR_weight_attribution='Alan Arazi et al., TabSTAR, https://huggingface.co/alana89/TabSTAR; model card CC-BY-4.0, README says MIT; retain stricter CC-BY attribution and both source records',
                   E5_attribution='intfloat/e5-small-v2, https://huggingface.co/intfloat/e5-small-v2, model card MIT',
                   classifiers_fitted=0, GPU_used=False, credentials_created=False)
    (ROOT/'research_r79_download_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    state('all_pinned_files_verified_no_model_execution', files=len(records))

if __name__ == '__main__':
    try: main()
    except BaseException as error:
        state('failed', error_type=type(error).__name__)
        raise SystemExit(1)
