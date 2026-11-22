"""Single sequential GPU owner: missing concat-C only, then checkpoint re-evaluation."""
import os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
project=root/'assignment_2b_emotion'
env=dict(os.environ,PYTHONPATH=str(project/'src'),HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
log=project/'Results/recovery_verification.log'
with log.open('a',encoding='utf-8') as handle:
 def run(args):
  print('RUN '+' '.join(args),flush=True)
  subprocess.run([sys.executable,'-u','-B',*args],cwd=project,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
 if not (project/'Results/per_fold/fusion_concat_protoC/summary.json').exists():
  run(['-m','mer.training.p0_train','--mode','fusion','--config','configs/fusion_concat.yaml','--protocol','C'])
 run(['-m','mer.evaluation.reconcile_results'])
print('Model verification complete',flush=True)
