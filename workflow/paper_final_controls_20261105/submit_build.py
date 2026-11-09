"""Submit a checked isolated document build, without waiting for completion."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    target=ROOT/'build_submission.json';assert not target.exists()
    v=json.loads((ROOT/'verification.json').read_text());assert v['passed']
    assert sha(v['archive'])==v['archive_sha256']
    subprocess.run(['bash','-n',str(ROOT/'build.pbs')],check=True)
    record=dict(released=False,reason='Login build interrupted with status 143; complete cross-reference convergence on a compute node.',
        hashes={str(p):sha(p) for p in (ROOT/'build.py',ROOT/'build.pbs',Path(v['archive']))})
    result=subprocess.run(['qsub','-h',str(ROOT/'build.pbs')],capture_output=True,text=True,check=True)
    jid=re.search(r'\b\d+\.gadi-pbs\b',result.stdout)[0]
    record['job_id']=jid;target.write_text(json.dumps(record,indent=2)+'\n')
    result=subprocess.run(['qstat','-xf','-F','json',jid],capture_output=True,text=True,check=True)
    job=json.loads(result.stdout)['Jobs'][jid];r=job['Resource_List']
    assert job['project']=='mk27' and job['queue'] in ('rsaa','rsaa-exec') and job['job_state']=='H'
    assert int(r['ncpus'])==16 and r['walltime']=='00:15:00'
    assert str(r['mem']).lower() in ('16gb',str(16*1024**3)+'b')
    record['effective_job_before_release']=job;target.write_text(json.dumps(record,indent=2)+'\n')
    subprocess.run(['qrls',jid],capture_output=True,text=True,check=True)
    record['released']=True;target.write_text(json.dumps(record,indent=2)+'\n')
    print('Document build checked and released:',jid,'; no completion wait.',flush=True)


if __name__=='__main__':main()
