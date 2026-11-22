"""Generate final evaluation only from checkpoint-linked verification outputs."""
from pathlib import Path
import os,sys,json,shutil,hashlib,subprocess
root=Path(__file__).resolve().parents[1];project=root/'assignment_2b_emotion';os.chdir(project);sys.path.insert(0,str(project/'src'))
os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['HF_HUB_OFFLINE']='1';os.environ['NUMBA_NUM_THREADS']='2'
import torch,pandas as pd,numpy as np
torch.set_num_threads(2)
from mer.evaluation.validate_evaluation import validate_run
from mer.evaluation.aggregate_results import aggregate
from mer.evaluation.significance import fusion_help,mcnemar_test
from mer.training.runner import run_test
from mer.evaluation.reconcile_results import sha
verified=Path('Results/verified_p0');plots=verified/'plots';plots.mkdir(exist_ok=True)
required=[f'{model}_proto{p}' for model in ['speech_wavlm_bigru_asp','text_roberta_top2ft','fusion_concat','fusion_gated_xattn'] for p in 'ABC']
assert all((verified/n/'verification.json').is_file() for n in required)
for n in required:validate_run(verified/n)
# Preserve auditable baseline artifacts; these models were not serialized historically.
for d in Path('Results/per_fold').glob('classical_*'):
 validate_run(d);dest=verified/d.name;dest.mkdir(exist_ok=True)
 for name in ['summary.json','predictions.csv']:
  if not (dest/name).exists():shutil.copy2(d/name,dest/name)
 (dest/'baseline_provenance.json').write_text(json.dumps(dict(source=str(d),fit_policy='train plus validation; differs from neural train-only fitting',prediction_sha256=sha(dest/'predictions.csv'),note='Existing predictions independently validated; no historical fitted-estimator checkpoint exists.'),indent=2))
# Histories are descriptive; verify their selected epoch against the final weights.
for n in required:
 history=Path('Results/per_fold')/n/'train_history.json'
 if not history.exists():continue
 h=json.loads(history.read_text());validated=[]
 for key,rows in h.items():
  ck=Path('checkpoints')/n/(key+'.pt');state=torch.load(ck,map_location='cpu',weights_only=True,mmap=True)
  ep=state['epoch'];row=next((r for r in rows if r['epoch']==ep),None)
  if row is None or abs(row['val_f1']-state['val_f1'])>1e-5:raise ValueError('History/checkpoint mismatch '+str(ck))
  validated.append(key);del state
 shutil.copy2(history,verified/n/'train_history.json')
 (verified/n/'history_validation.json').write_text(json.dumps(dict(source=str(history),selected_epochs_match=validated,note='Legacy curves checked against selected checkpoint epoch/F1; complete training lineage is not reconstructed.'),indent=2))
for p in 'AB':
 n=f'fusion_gated_xattn_proto{p}';out=verified/(n+'_zerotext')
 if not (out/'summary.json').exists():
  before={str(x):sha(x) for x in (Path('checkpoints')/n).glob('fold_*.pt')}
  run_test('fusion','configs/fusion.yaml',p,zero_text=True,save_tag=str(out.resolve()))
  assert before=={str(x):sha(x) for x in (Path('checkpoints')/n).glob('fold_*.pt')}
  (out/'ablation_provenance.json').write_text(json.dumps(dict(parent=n,checkpoint_sha256=before,intervention='zero text representations at intermediate fusion projections; not a separately trained speech system'),indent=2))
 validate_run(out)
aggregate(verified,verified)
from mer.evaluation import plots as plot
plot.RUN_ROOT=verified;plot.PLOTS=plots
for n in required:
 _,d=validate_run(verified/n)
 plot.plot_confusion(n,n.replace('_',' '),f'cm_{n}.png')
for n,title,out in [('speech_wavlm_bigru_asp_protoA','Speech, word-disjoint','cm_speech_protoA.png'),('speech_wavlm_bigru_asp_protoB','Speech, speaker-independent','cm_speech_protoB.png'),('text_roberta_top2ft_protoA','Text, word-disjoint','cm_text_protoA.png'),('fusion_gated_xattn_protoA','Gated fusion, word-disjoint','cm_fusion_protoA.png'),('fusion_gated_xattn_protoB','Gated fusion, speaker-independent','cm_fusion_protoB.png')]:plot.plot_confusion(n,title,out)
plot.plot_model_comparison();plot.plot_layer_weights('speech_wavlm_bigru_asp_protoA')
plot.plot_training_curves({'Speech':'speech_wavlm_bigru_asp_protoA','Text':'text_roberta_top2ft_protoA','Gated fusion':'fusion_gated_xattn_protoA'})
shutil.copy2('Results/plots/data_audit_distributions.png',plots/'data_audit_distributions.png')
blocks=[('Temporal (speech)','speech_wavlm_bigru_asp_protoA'),('Contextual (text)','text_roberta_top2ft_protoA'),('Fusion','fusion_gated_xattn_protoA')]
for method in ['tsne','umap']:plot.plot_embeddings_panel(blocks,method=method,out_name=f'embeddings_{method}_protoA.png')
subprocess.run([sys.executable,'-B','-m','mer.evaluation.failure_cases','--root',str(verified),'--out_md',str(verified/'failure_cases.md'),'--out_plot',str(plots/'failure_cases.png')],check=True)
lines=['# Prediction-backed comparative analysis','', 'All scores use frozen test IDs and final checkpoint-linked re-evaluation. Changes are descriptive; with only two speakers no population-generalization claim is justified.','', '## Per-emotion recall and fusion effects']
for p in 'ABC':
 frames={m:validate_run(verified/f'{m}_proto{p}')[0] for m in ['speech_wavlm_bigru_asp','fusion_concat','fusion_gated_xattn']}
 speech=frames['speech_wavlm_bigru_asp']
 for m in ['fusion_concat','fusion_gated_xattn']:
  delta=fusion_help(speech,frames[m]);delta.to_csv(verified/f'{m}_proto{p}_help_harm.csv',index=False)
  stats=mcnemar_test(speech,frames[m]);(verified/f'{m}_proto{p}_paired.json').write_text(json.dumps(stats,indent=2))
  lines.append(f"- Protocol {p}, {m}: helped {int(delta.helped.sum())}, harmed {int(delta.harmed.sum())} utterances versus speech. Exploratory paired statistics ignore within-word/speaker and CV dependence.")
  by=delta.groupby('emotion').accuracy_delta.mean().sort_values(ascending=False)
  lines.append('  Per-emotion accuracy changes (percentage points): '+', '.join(f'{k} {v*100:+.2f}' for k,v in by.items()))
 for m,df in frames.items():
  rec=df.assign(correct=df.y_true==df.y_pred).groupby('true_label').correct.mean().sort_values()
  lines.append(f"- Protocol {p}, {m}: lowest recall {rec.index[0]}={rec.iloc[0]*100:.2f}%; highest {rec.index[-1]}={rec.iloc[-1]*100:.2f}%. Ties are not evidence of a unique easiest emotion.")
lines+=['','## Interpretation','Speech alone exceeds gated fusion in aggregate A/B/C. Text is label-independent in the complete balanced corpus, so fusion improvement is not presumed. The B drop is a two-speaker transfer stress test, not a reliable population estimate. Measured F0/duration in the five failures are descriptive; they do not establish causal explanations without controlled tests or listening-based reference judgments.','', 'The classical baselines fit train+validation, whereas neural models fit train only. Matched-data baseline diagnostics are retained under Results/audits/baseline_recovery and should be used when discussing fairness. Zero-percent random-split TF-IDF accuracy is reproduced by complementary per-transcript class imbalance, not evidence of swapped labels. A 14.29% accuracy reference is not a universal macro-F1 threshold.','', 'Gate values summarize the mixture of two already cross-attended streams; they are not causal modality contributions. Zero-text results test an intermediate intervention, not a retrained speech-only model. The performance drop under that intervention must be reported, not asserted to match speech-only.','', 'Gated Protocol C has no saved training history. Its final checkpoints and test inference are verified; no missing epoch curve was fabricated. Interrupted late fusion is excluded from headline results and was not restarted. Historical text-training lineage remains uncertain, but the final checkpoint-to-prediction mapping is now reproducible and hash-linked.']
(verified/'comparative_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
# Publish required deliverables, preserving all legacy per-fold runs.
for name in ['accuracy_tables.csv','accuracy_tables.md','per_emotion_validated.csv','evaluation_validation.json','failure_cases.csv','failure_cases.md','comparative_analysis.md']:
 shutil.copy2(verified/name,Path('Results')/name)
for p in plots.glob('*'):
 if p.is_file() and p.name!='data_audit_distributions.png':shutil.copy2(p,Path('Results/plots')/p.name)
(verified/'reconciliation_status.json').write_text(json.dumps(dict(status='FINAL_CHECKPOINT_INFERENCE_RECONCILED',headline_runs=required,incident_sha256=sha('Results/audits/process_collision/incident.json'),note='Historical incident is retained unchanged; invalidated legacy P0 completion markers remain unusable.'),indent=2))
print('Final evaluation deliverables generated from verified predictions',flush=True)
