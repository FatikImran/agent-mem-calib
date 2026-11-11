"""Run approved fresh passes 2 and 3 sequentially, with durable logs and scoring."""
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime
import msvcrt


def main():
    root=Path(__file__).resolve().parents[1]
    gpu=root/'data/runtime/generator-cuda/Scripts/python.exe'
    retrieval=root/'data/runtime/retrieval/Scripts/python.exe'
    lock_path=root/'data/retrieval-dev/approved-repeats.lock'
    with lock_path.open('a+b') as lock:
        if lock.tell()==0:
            lock.write(b'0');lock.flush()
        lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise RuntimeError('Another overnight-repeat launcher is already active')
        try:
            for repeat in (2,3):
                folder=root/f'results/repeat-{repeat}'
                folder.mkdir(parents=True,exist_ok=True)
                log_path=folder/'execution.log'
                def run(command):
                    with log_path.open('a',encoding='utf8') as log:
                        heading=f'\n{datetime.now().astimezone().isoformat()} {command}\n'
                        print(heading,flush=True);log.write(heading);log.flush()
                        with subprocess.Popen(command,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace') as proc:
                            for line in proc.stdout:
                                print(line,end='',flush=True);log.write(line);log.flush()
                            code=proc.wait()
                        if code:raise RuntimeError(f'Command failed ({code}); see {log_path}. Rerun launcher to resume.')
                run([str(retrieval),'-u','-m','scripts.prepare_repeat_gate','--repeat',str(repeat),'--offline','--development','data/retrieval-dev/eval-paired.json'])
                original=json.loads((root/'results/first-pass/memgate-heldout100-development.json').read_bytes())
                fresh=json.loads((folder/'memgate-heldout100-development.json').read_bytes())
                if original['rows']!=fresh['rows']:
                    raise RuntimeError('Fresh gate scores differ from pass one; preserve outputs and review before continuing.')
                run([str(gpu),'-u','-m','scripts.run_approved_repeat','--approved-evaluation','--repeat',str(repeat)])
                run([str(gpu),'-u','-m','scripts.score_primary_repeat','--repeat',str(repeat)])
            print('Both fresh repeat passes completed and scored. Safe to close this window.',flush=True)
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


if __name__=='__main__':main()
