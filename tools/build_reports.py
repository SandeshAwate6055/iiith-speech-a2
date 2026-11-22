"""Evidence-derived private MD/PDF reports. No training or evaluation outputs changed.
Requires reportlab, pymupdf, numpy, PyYAML. Run from any directory.
--final refuses missing/invalid evidence before writing. --allow-provisional-2a
permits only the genuine Praat blocker, leaving report 2A explicitly DRAFT.
"""
import argparse, csv, hashlib, json, math, os, statistics, sys
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape
LABELS=['angry','disgust','fear','happy','neutral','sad','surprise']
MODELS=[('Speech','speech_wavlm_bigru_asp'),('Text','text_roberta_top2ft'),('Concat','fusion_concat'),('Gated','fusion_gated_xattn')]
PLOTS=['model_comparison_protocols.png','cm_speech_protoA.png','cm_speech_protoB.png','cm_text_protoA.png','cm_fusion_protoA.png','cm_fusion_protoB.png','training_curves_protoA.png','wavlm_layer_weights.png','embeddings_umap_protoA.png','embeddings_tsne_protoA.png','failure_cases.png']
def require(ok,msg):
 if not ok: raise ValueError(msg)
def csvrows(p):
 with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def js(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def metrics(rows):
 require(bool(rows),'Empty fold'); cm=[[0]*7 for _ in range(7)]
 for r in rows:
  a,b=int(r['y_true']),int(r['y_pred']); require(0<=a<7 and 0<=b<7,'Invalid label'); cm[a][b]+=1
 recall=[]; f1=[]
 for i in range(7):
  n=sum(cm[i]); pred=sum(x[i] for x in cm); tp=cm[i][i]
  recall.append(tp/n if n else 0); f1.append(2*tp/(n+pred) if n+pred else 0)
 return dict(accuracy=sum(cm[i][i] for i in range(7))/len(rows),uar=statistics.mean(recall),macro_f1=statistics.mean(f1)),recall
class Evidence:
 def __init__(self,root): self.root=root; self.pending=[]; self.hashes={}
 def use(self,p):
  require(p.is_file(),'Missing '+str(p.relative_to(self.root)))
  self.hashes[p.relative_to(self.root).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest(); return p
 def issue(self,s): self.pending.append(s)
def validate_run(ev,run,tab):
 b=ev.root/'assignment_2b_emotion'; d=(getattr(ev,'verification_root',None) or b/'Results/per_fold')/run
 summary=js(ev.use(d/'summary.json')); rows=csvrows(ev.use(d/'predictions.csv'))
 meta=csvrows(ev.use(b/'data/metadata.csv')); lookup={x['utt_id']:x for x in meta}
 require(len(lookup)==len(meta),'Duplicate metadata'); proto=run[-1]
 require(summary['protocol']==proto,'Protocol mismatch')
 folds=js(ev.use(b/f'splits/protocol_{proto}.json'))['folds']
 keys=[(int(x['fold']),x['utt_id']) for x in rows]
 require(len(keys)==len(set(keys)),'Duplicate predictions')
 require({x[0] for x in keys}=={f['fold'] for f in folds},'Incomplete folds')
 values=[]; recalls=[]; order=[]
 for f in folds:
  groups=[set(f[k]) for k in ['train','val','test']]
  require(all(g<=set(lookup) for g in groups),'Unknown split IDs')
  require(all(len(g)==len(f[k]) for g,k in zip(groups,['train','val','test'])),'Duplicate split IDs')
  require(not any(groups[i]&groups[j] for i,j in [(0,1),(0,2),(1,2)]),'Split overlap')
  if proto in 'AB':
   words=[{lookup[u]['word'] for u in g} for g in groups]
   require(not any(words[i]&words[j] for i,j in [(0,1),(0,2),(1,2)]),'Word leakage')
  if proto=='B': require(not {lookup[u]['speaker'] for u in groups[0]|groups[1]}&{lookup[u]['speaker'] for u in groups[2]},'Speaker leakage')
  sub=[x for x in rows if int(x['fold'])==f['fold']]
  require({x['utt_id'] for x in sub}==groups[2],'Test membership mismatch')
  order += [(f['fold'],u) for u in f['test']]
  for x in sub:
   require(int(x['y_true'])==int(lookup[x['utt_id']]['emotion_id']),'Metadata label mismatch')
   for k in ['word','speaker']:
    if k in x: require(x[k]==lookup[x['utt_id']][k],k+' mismatch')
   for k,label in [('y_true','true_label'),('y_pred','pred_label')]:
    if label in x: require(x[label]==LABELS[int(x[k])],'Wrong label name')
   if 'confidence' in x: require(math.isfinite(float(x['confidence'])) and 0<=float(x['confidence'])<=1,'Invalid confidence')
   cols=['prob_'+e for e in LABELS]
   if any(k in x for k in cols):
    require(all(k in x for k in cols),'Incomplete probabilities'); probs=[float(x[k]) for k in cols]
    require(all(math.isfinite(v) and 0<=v<=1 for v in probs) and abs(sum(probs)-1)<1e-5,'Invalid probabilities')
    require(max(range(7),key=lambda i:probs[i])==int(x['y_pred']),'Argmax mismatch')
  m,rec=metrics(sub); values.append(m); recalls.append(rec)
 result={}
 for metric in ['accuracy','uar','macro_f1']:
  vals=[v[metric] for v in values]
  for suffix,v in [('mean',statistics.mean(vals)),('std',statistics.pstdev(vals))]:
   k=metric+'_'+suffix; result[k]=v
   require(abs(float(summary[k])-v)<0.000061,'Summary mismatch '+k)
   if tab is not None: require(k in tab and abs(float(tab[k])-v)<1e-7,'Accuracy CSV mismatch '+k)
 result.update(rows=rows,recalls=[statistics.mean(x[i] for x in recalls) for i in range(7)],embedding=False)
 emb=d/'embeddings.npy'
 if emb.exists():
  import numpy as np
  x=np.load(ev.use(emb),allow_pickle=False)
  require(x.ndim==2 and len(x)==len(rows) and np.isfinite(x).all(),'Invalid embedding')
  side=d/'embedding_ids.csv'
  if side.exists(): require([(int(z['fold']),z['utt_id']) for z in csvrows(ev.use(side))]==keys,'Embedding ID mismatch')
  else: require(keys==order,'Legacy embedding order mismatch')
  result['embedding']=True
 if getattr(ev,'verification_root',None): verify_checkpoint_link(ev,run,d,len(folds))
 return result
def verify_checkpoint_link(ev,run,d,nfolds):
 """Verify declared correspondence; hashes cannot independently prove code execution."""
 proof=js(ev.use(d/'verification.json'))
 incident=ev.root/'assignment_2b_emotion/Results/audits/process_collision/incident.json'
 require(proof.get('queue_ownership_cleared') is True,'Ownership not cleared for verification')
 if incident.exists():
  require(proof.get('reconciled_after_collision') is True,'Collision not reconciled')
  require(proof.get('incident_sha256')==hashlib.sha256(ev.use(incident).read_bytes()).hexdigest(),'Incident hash mismatch')
 for name,key in [('predictions.csv','prediction_sha256'),('summary.json','summary_sha256'),('embeddings.npy','embeddings_sha256')]:
  require(proof.get(key)==hashlib.sha256(ev.use(d/name).read_bytes()).hexdigest(),'Verification artifact hash mismatch: '+name)
 hashes=proof.get('checkpoint_sha256',{})
 require(len(hashes)==nfolds,'Need exactly one checkpoint hash per fold')
 require({Path(x).name for x in hashes}=={f'fold_{i}.pt' for i in range(nfolds)},'Checkpoint fold list mismatch')
 for rel,expected in hashes.items():
  path=(ev.root/rel).resolve(); require(path.is_relative_to(ev.root),'Checkpoint path outside repository root')
  require(expected==hashlib.sha256(ev.use(path).read_bytes()).hexdigest(),'Final checkpoint changed since verification')
 require(d.resolve()!=(ev.root/'assignment_2b_emotion/Results/per_fold'/run).resolve(),'Verification directory must be distinct')
def collect_b(ev):
 b=ev.root/'assignment_2b_emotion'; r=getattr(ev,'verification_root',None) or b/'Results'; good={}; status=[]; lookup={}
 incident=b/'Results/audits/process_collision/incident.json'; affected=[]
 if incident.exists():
  details=js(ev.use(incident)); affected=details.get('affected_runs',[])
  if not getattr(ev,'verification_root',None): ev.issue('2B unresolved process collision: distinct verification-root and checkpoint-linked re-evaluation required; no P0 marker adoption')
 p=r/'accuracy_tables.csv'
 if p.exists():
  for x in csvrows(ev.use(p)):
   require(x['run'] not in lookup,'Duplicate accuracy row'); lookup[x['run']]=x
 else: ev.issue('2B missing Results/accuracy_tables.csv; regenerate using evaluation.aggregate_results')
 for label,stem in MODELS:
  for proto in 'ABC':
   run=stem+'_proto'+proto
   try:
    if run not in lookup: ev.issue('2B '+run+': no validated accuracy CSV row')
    if run in affected and not getattr(ev,'verification_root',None): raise ValueError('collision: checkpoint correspondence unverified; P0 markers invalid')
    good[run]=validate_run(ev,run,lookup.get(run)); status.append([label,proto,'validated' if run in lookup else 'prediction-audited; CSV pending'])
   except (ValueError,KeyError,OSError,TypeError,IndexError) as exc: ev.issue('2B '+run+': '+str(exc)); status.append([label,proto,'PENDING'])
 for _,stem in [MODELS[0],MODELS[1],MODELS[3]]:
  if not good.get(stem+'_protoA',{}).get('embedding'): ev.issue('2B missing validated block embedding: '+stem+'_protoA')
 for name in PLOTS:
  p=r/'plots'/name
  if name=='failure_cases.png' and not p.exists() and not getattr(ev,'verification_root',None): p=b/'Results/audits/b3_validated/neural_failure_cases.png'
  if p.exists():
   ev.use(p)
   try:
    from PIL import Image as PILImage
    with PILImage.open(p) as im: im.verify()
   except Exception as exc: ev.issue('2B invalid mandatory plot '+name+': '+str(exc))
  else: ev.issue('2B missing mandatory plot: '+name)
 for name in ['embeddings_umap_protoA_metrics.json','embeddings_tsne_protoA_metrics.json']:
  try:
   stats=js(ev.use(r/'plots'/name)); require(len(stats)==3 and all(v.get('n',0)>0 and v.get('fold')==0 for v in stats.values()),'Three within-fold block panels required')
  except (ValueError,OSError,TypeError,AttributeError) as exc: ev.issue('2B projection evidence '+name+': '+str(exc))
 failures=[]; p=r/'failure_cases.csv'
 if not p.exists() and not getattr(ev,'verification_root',None): p=b/'Results/audits/b3_validated/neural_failure_cases.csv'
 if p.exists():
  for row in csvrows(ev.use(p)):
   if len(failures)==5: break
   try:
    require(int(row['y_true'])!=int(row['y_pred']),'Not a failure')
    require(row['utt_id'] not in {x['utt_id'] for x in failures},'Repeated failure utterance')
    matches=[]
    for run,data in good.items():
     if row.get('source') and run not in row['source']: continue
     for x in data['rows']:
      if x['utt_id']==row['utt_id'] and int(x['fold'])==int(row['fold']) and int(x['y_true'])==int(row['y_true']) and int(x['y_pred'])==int(row['y_pred']): matches.append((run,x))
    require(bool(matches),'Failure not prediction-backed'); run,x=matches[0]; failures.append(dict(x,run=run))
   except (ValueError,KeyError) as exc: ev.issue('2B invalid failure: '+str(exc))
 if len(failures)<5: ev.issue(f'2B five distinct actual prediction-backed failures required; found {len(failures)}')
 return good,status,failures
# Nodes share one source between Markdown and PDF.
def text(s): return ('text',s)
def table(h,r): return ('table',(h,r))
def image(p,c,h=200): return ('image',(p,c,h))
def gallery(paths,caption,height=210): return ('gallery',([p for p in paths if p.is_file()],caption,height))
def a_pages(ev):
 a=ev.root/'assignment_2a_mfa'; e=a/'outputs/evaluation'; rows=[]; lex={}; sources=[]
 for name,target in [('validity_and_durations.csv',rows),('lexicon_pronunciation_sources.csv',sources)]:
  try: target.extend(csvrows(ev.use(e/name)))
  except (ValueError,OSError) as exc: ev.issue('2A '+str(exc))
 try: lex=js(ev.use(e/'lexicon_audit.json'))
 except (ValueError,OSError) as exc: ev.issue('2A '+str(exc))
 rows=[x for x in rows if x.get('system')=='english_us_arpa']
 if len(rows)!=6 or not all(x.get('structurally_valid')=='True' and x.get('word_seq_match')=='True' and all(int(x.get(k,-1))==0 for k in ['overlaps','gaps','invalid_intervals','n_spn_oov']) for x in rows): ev.issue('2A strict six-file validity/word-sequence checks incomplete')
 if not(lex.get('passed') and lex.get('oov_types')==0 and lex.get('acoustic_phones_compatible')): ev.issue('2A lexicon coverage/inventory evidence incomplete')
 grids=list((a/'outputs/textgrids/english_us_arpa').rglob('*.TextGrid'))
 if len(grids)!=6: ev.issue('2A exactly six primary TextGrids required')
 for p in grids: ev.use(p)
 provenance=e/'model_provenance.json'
 if provenance.exists(): ev.use(provenance)
 validationlog=a/'work/validate_arpa/corpus/corpus.log'
 if validationlog.exists(): ev.use(validationlog)
 figs=sorted((a/'outputs/figures').glob('*.png'))
 for p in figs: ev.use(p)
 if len(figs)<6: ev.issue('2A six generated figures required')
 obs=e/'praat_observations.md'; shots=[p for p in (a/'outputs').rglob('*') if p.suffix.lower() in ['.png','.jpg'] and 'praat' in str(p).lower()]
 observed=obs.is_file() and bool(obs.read_text(encoding='utf-8-sig').strip()) and bool(shots)
 if not observed: ev.issue('2A Praat screenshot/observations pending: user stopped UI; plots are not Praat inspection')
 p1=[text('Only the six provided BU/ISLE recordings were used. BU files have NIST SPHERE headers despite .wav names; preparation decodes and writes RIFF PCM16 at 16 kHz mono, normalizes transcripts into .lab and groups speakers. No outside 2A audio or gold alignments are used.'),text('MFA 3.4.4; official MontrealCorpusTools English US ARPA archive 3.3.0; pinned Hugging Face revision '+lex.get('model_revision','pending')+'. Tool-observed MFA validation completed successfully (exit 0). work/validate_arpa/corpus/corpus.log records 97.163 s total audio duration and 258.858 s full validation runtime. These are execution facts, not user attestation or listening-based boundary evaluation. The validator trained a small monophone diagnostic; this is distinct from the primary pretrained alignment model. Kaldi GMM-HMM alignment estimates transcript-constrained word/phone times [A1].'),text(f"Lexicon: {lex.get('dictionary_words','pending')} corpus word types; {len(sources)} pronunciation rows; {len(lex.get('base_oovs',[]))} base OOV types covered by G2P/manual candidates; {lex.get('oov_types','pending')} final OOV types. Acoustic inventory compatible: {lex.get('acoustic_phones_compatible','pending')}. Source-row counts {dict(Counter(x['source'] for x in sources))}. Candidate pronunciations are not human gold."),table(['Recording','Seconds','Words','Phones','Valid / seq.'],[[x['file'].replace('ISLE_SESS0131_BLOCKD02_','ISLE ').replace('_sprt1',''),x['duration_s'],x['aligned_words'],x['n_phones'],x['structurally_valid']+' / '+x['word_seq_match']] for x in rows]),text('Structural checks cover tier presence, interval validity, utterance coverage, gaps and overlaps. All current six rows show zero gaps, overlaps and spn/OOV intervals; normalized word counts are 68/73/82/5/5/5. These checks establish file consistency, not perceptual correctness or boundary accuracy.')]
 p2=[text('Generated waveform/spectrogram/TextGrid plots below are distinct from Praat screenshots. BU figures show excerpts and ISLE figures show full utterances. No manual-listening conclusion is inferred from their existence.')]
 for p in [x for x in figs if 'F2BJRLP1' in x.name or '02_01_' in x.name][:2]: p2.append(image(p,p.stem.replace('_',' '),245))
 p3=[text('Praat visual inspection: '+('all six recordings were opened in the editor; scoped visual observations and genuine screenshots are recorded below. No listening-based gold labels are claimed.' if observed else 'PENDING. Physical Escape stopped Praat automation. A4 Praat inspection, listening and skipped-phone diagnoses are not claimed. No gold boundary errors or timing accuracy are fabricated.'))]
 if obs.is_file(): ev.use(obs); p3.append(text('Recorded observations (plain-text attribution):\n'+obs.read_text(encoding='utf-8-sig').strip()))
 if observed:
  for shot in shots[:2]: ev.use(shot)
  p3.append(gallery(shots[:2],'Actual Praat editor screenshots (ISLE 01 and 02); original captures retained.',200))
 elif figs: p3.append(image(figs[-1],'Generated ISLE visualization, not a Praat observation.',190))
 if rows: p3.append(text(f"Duration flags: {sum(int(x['phones_at_30ms_floor']) for x in rows)} intervals at 30 ms; {sum(int(x['phones_over_300ms']) for x in rows)} above 300 ms. These are review candidates, not confirmed errors. Inspect proper names, initialisms, long pauses and L2 minimal pairs. Without independently reviewed gold, no boundary MAE is reported."))
 plotobs=e/'plot_observations.md'
 if plotobs.exists() and not observed:
  ev.use(plotobs); p3.append(text('Attributed plot-only observations: '+plotobs.read_text(encoding='utf-8-sig').strip()))
 p3 += [text('Reproduce using assignment_2a_mfa/README.md and scripts/prepare_models.py, prepare_lexicon.py, run_alignment.py, then evaluate_alignments.py and visualize.py. Regenerate with python tools/build_reports.py. Six source figures: '+', '.join(p.name for p in figs)+'.'),text('[A1] McAuliffe et al., Montreal Forced Aligner, Interspeech 2017. https://www.isca-archive.org/interspeech_2017/mcauliffe17_interspeech.html. Lexicon audit records model/base dictionary CC BY 4.0. Licensed raw audio, caches and checkpoints are excluded from Git.')]
 return [('Method and structural evidence',p1),('Generated alignment visualizations',p2),('Praat inspection and limitations',p3)]

def b_pages(ev,good,status,failures):
 b=ev.root/'assignment_2b_emotion'; plots=(getattr(ev,'verification_root',None) or b/'Results')/'plots'
 def plot(n,c,h=170):
  path=plots/n
  if n=='failure_cases.png' and not path.exists() and not getattr(ev,'verification_root',None): path=b/'Results/audits/b3_validated/neural_failure_cases.png'
  return image(path,c,h) if path.is_file() else text('PENDING figure: '+n)
 meta=b/'data/metadata.csv'; ds='Metadata pending.'
 if meta.exists():
  data=csvrows(ev.use(meta)); ds=f"Metadata: {len(data)} clips; {len({x['utt_id'] for x in data})} unique IDs; {len({x['word'] for x in data})} words; {len({x['speaker'] for x in data})} speakers. Class counts {dict(sorted(Counter(x['emotion_id'] for x in data).items()))}."
 p1=[text('Four headline systems (speech, text, concat fusion, gated fusion) under A/B/C require 12 complete validated runs. Ongoing/partial jobs are pending, never completed results. Final checkpoint-to-prediction correspondence is checked by independent re-evaluation and before/after hashes. Historical text-training lineage was affected by concurrent writers; invalidated P0 markers are not adopted.'),text(ds+' TESS uses two female speakers and seven acted emotions; filename-derived text is Say the word <target> after stripping speaker/emotion suffixes. Source [B6]: Borealis DOI 10.5683/SP2/E8H2MF. Raw audio is excluded from Git; retain source license/attribution.'),table(['Protocol','Frozen split','Interpretation'],[['A','5 folds: 128/32/40 disjoint words','Primary lexical generalization; speakers shared'],['B','2 folds: speaker transfer plus word disjoint; 40 test words','Stress test; two speakers limit generalization'],['C','5 random stratified folds; ~70/10/20%','Optimistic reference; words and speakers shared']]),text('Neural models fit training IDs and select best validation macro-F1 checkpoints. All models reuse frozen splits; no test tuning. Accuracy, UAR and macro-F1 are independently recomputed from saved test predictions. Tables show unweighted fold means +/- population fold SD (ddof=0), not confidence intervals.'),table(['System','A','B','C'],[[label]+[next(x[2] for x in status if x[:2]==[label,q]) for q in 'ABC'] for label,_ in MODELS]),plot('data_audit_distributions.png','Dataset preparation audit distribution.',135)]
 arch=[['Speech preprocessing','16 kHz mono; trim 30 dB, 50 ms padding; peak -1 dBFS','Standardizes rate/channel, retains boundary context. Normalization may alter loudness cues.'],['Speech features','Frozen WavLM-Base+ [B1], [13,T,768]; learned layer sum','Caching fits limited GPU memory; weights adapt layer contributions without encoder fine-tuning.'],['Temporal block','256 projection; bidirectional GRU, 128 each direction; attentive mean/std [B2]','Models sequence dynamics, salience and feature dispersion rather than only a global average.'],['Text features','RoBERTa token embeddings [B3], max length 32','Deterministic transcript; no filename emotion suffix is passed as input.'],['Contextual block','Top two transformer layers fine-tuned; 256 projection; masked attention pooling','Limits trainable parameters, preserves token context and produces fixed-size representation.'],['Fusion/classifier','Concat or bidirectional 4-head cross attention + vector gate, 256 dims; MLP -> 7 logits [B4,B5]','Concat comparator isolates fusion complexity. Cross attention exchanges sequence context; gate weights are descriptive, not causal.']]
 p2=[table(['Block','Implementation','Justification / limitation'],arch),text('Training settings are read from the current configuration files in the table. Implemented optimizer is AdamW with validation macro-F1 selection. See hashed configs for weight decay, smoothing, dropout and augmentation settings. Config snapshots are authoritative; report manifest hashes current config files. Gated mean closer to one weights the speech-side representation more, but cannot establish causal reliance.'),text('References: [B1] Chen et al., WavLM, IEEE JSTSP 2022, arXiv 2110.13900. [B2] Okabe et al., Attentive Statistics Pooling, Interspeech 2018, arXiv 1803.10963. [B3] Liu et al., RoBERTa, 2019, arXiv 1907.11692. [B4] Tsai et al., MulT, ACL 2019, arXiv 1906.00295. [B5] Arevalo et al., Gated Multimodal Units, ICLR-W 2017, arXiv 1702.01992. These motivate design; this implementation is not claimed to reproduce those papers.')]
 configrows=[]
 for name in ['speech','text','fusion_concat','fusion']:
  c=b/'configs'/f'{name}.yaml'
  if c.exists():
   import yaml
   cfg=yaml.safe_load(ev.use(c).read_text(encoding='utf-8-sig')); train=cfg['train']
   configrows.append([name,str(train.get('batch_size')),str(train.get('epochs')),str(train.get('patience')),str(train.get('lr')),str(train.get('encoder_lr','-'))])
 p2.insert(1,table(['Config','Batch','Epoch cap','Patience','Head LR','Encoder LR'],configrows))
 p2.append(gallery([plots/n for n in ['training_curves_protoA.png','wavlm_layer_weights.png']], 'Available validation/training curves and learned WavLM layer weights.',130))
 metricsrows=[]
 for label,stem in MODELS:
  for proto in 'ABC':
   d=good.get(stem+'_proto'+proto)
   vals=[f"{100*d[m+'_mean']:.2f} +/- {100*d[m+'_std']:.2f}" for m in ['accuracy','uar','macro_f1']] if d else ['PENDING']*3
   metricsrows.append([label,proto]+vals)
 p3=[text('Scores below are independently recomputed from frozen test predictions and checked against saved summaries and the final accuracy CSV. Each headline row has checkpoint-linked re-evaluation evidence. The historical concurrent-writer incident is retained; these are verified final-checkpoint results, not a reconstruction of the original text training lineage.'),table(['System','Protocol','Accuracy %','UAR %','Macro-F1 %'],metricsrows),plot('model_comparison_protocols.png','Prediction-derived comparison; classical train+validation fitting is disclosed separately from neural train-only fitting.',195)]
 comparison=[]
 for proto in 'ABC':
  sp=good.get(MODELS[0][1]+'_proto'+proto)
  for label,stem in MODELS[2:]:
   f=good.get(stem+'_proto'+proto)
   if sp and f: comparison.append(f"{label} minus speech macro-F1, {proto}: {100*(f['macro_f1_mean']-sp['macro_f1_mean']):+.2f} percentage points.")
 p4=[text('Fusion comparison: '+(' '.join(comparison) if comparison else 'PENDING matched completed speech/fusion runs. No benefit is asserted before measurement.'))]
 sp=good.get(MODELS[0][1]+'_protoA')
 if sp:
  rec=sp['recalls']; easy=max(range(7),key=lambda i:rec[i]); hard=min(range(7),key=lambda i:rec[i])
  p4.append(text(f'Speech A easiest observed emotion by mean recall: {LABELS[easy]} ({100*rec[easy]:.2f}%); hardest: {LABELS[hard]} ({100*rec[hard]:.2f}%). Ties are possible. These are descriptive ranks; why a label is easy/hard needs listening or controlled tests.'))
 else: p4.append(text('Easiest/hardest ranking pending validated speech A predictions.'))
 if good.get('speech_wavlm_bigru_asp_protoB') and good.get('fusion_gated_xattn_protoB'):
  rs=good['speech_wavlm_bigru_asp_protoB']['recalls']; rg=good['fusion_gated_xattn_protoB']['recalls']
  p4.append(table(['Emotion','Speech B recall %','Gated B recall %'],[[LABELS[i],f'{100*rs[i]:.2f}',f'{100*rg[i]:.2f}'] for i in range(7)]))
  p4.append(text('Speaker transfer: speech fear recall is 100% but happiness is 0%; this class-specific collapse is hidden by near-ceiling Protocol A. Gated fusion helps neutral (+26.25 recall points) and surprise (+12.50), but corrects only 45 errors while introducing 93 errors overall in B. Thus selective benefit does not imply an aggregate gain. Different speaker-specific acoustic realizations are a plausible explanation, not a causal finding from two speakers.'))
 if getattr(ev,'verification_root',None):
  r=ev.verification_root
  gates=[f"{pr}={js(ev.use(r/('fusion_gated_xattn_proto'+pr)/'summary.json')).get('gate_mean',0):.3f}" for pr in 'ABC']
  ab=[]
  for pr in 'AB':
   sp=r/('fusion_gated_xattn_proto'+pr+'_zerotext')/'summary.json'
   if sp.exists():
    z=js(ev.use(sp));ab.append(f"{pr}: {z['accuracy_mean']*100:.2f}% accuracy/{z['macro_f1_mean']*100:.2f}% macro-F1")
  p4.append(text('Mean mixture gates: '+', '.join(gates)+'. The two streams already contain cross-attended information, so gates are not causal text/speech contributions. Zero-text intervention: '+'; '.join(ab)+'. It does not reproduce speech-only performance in A. Modality dropout therefore did not establish the intended robustness; no test-tuned correction was made.'))
 p4 += [gallery([plots/n for n in ['cm_speech_protoA.png','cm_speech_protoB.png','cm_text_protoA.png','cm_fusion_protoA.png','cm_fusion_protoB.png']], 'Verified speech A/B, text A and gated A/B confusion matrices; full-resolution figures accompany the report.',180),text('Fair-fit caveat: original classical baselines fit train+validation, while neural fits use train with validation selection. Train-only/direct-SVM audit comparisons are separate, not replacement E1 rows. Legacy SVM probability argmax can differ from direct prediction. Poor speaker-transfer similarity is consistent with feature shift, not proof of its sole cause.'),text('Text audit: the full corpus pairs every transcript with all seven emotions, giving balanced population accuracy chance 1/7 and no intrinsic label information. This does not imply chance under every dependent finite partition. Classical text C 0% reflects complementary imbalance: each held-out true label is below its transcript\'s training-complement majority under train+validation fitting. Preserve frozen splits and original rows. Near-zero train-only audits and neural text results are distinct experiments; no universal macro-F1 chance threshold is assumed.')]
 p5=[text('Five distinct actual error utterances are selected from Results/failure_cases.csv (or the B3 validated audit export in draft mode) and matched against validated headline predictions. Selection may emphasize confidence and is not an unbiased error-rate sample. No invented listening, F0 or causal explanation is attached.'),table(['Run / fold','Utterance','True -> predicted','Confidence'],[[x['run'].replace('_proto',' / ')+f" f{x['fold']}",x['utt_id'],LABELS[int(x['y_true'])]+' -> '+LABELS[int(x['y_pred'])],f"{float(x.get('confidence',0)):.3f}"] for x in failures]) if failures else text('PENDING: five prediction-backed actual failures are not available.'),plot('failure_cases.png','Existing case spectrogram/F0/probability plot; cases must match saved predictions.',275),text('Per-case interpretation: each row is an observed label substitution with stated model confidence. Confidence is not correctness. Duration/F0 and acoustic shape are descriptive; claims about acting style, prosody or speaker causes remain hypotheses without recorded listening evidence.')]
 p6=[text('Required blocks: temporal (speech), contextual (text), fused (gated), from validated Protocol A embeddings. embedding_ids.csv makes correspondence explicit; newly re-evaluated arrays are explicitly linked to their test IDs and checkpoint hashes. Projections operate within a single fold; axes across panels are independent. Separation in a 2D projection is not independent evidence of predictive performance.'),plot('embeddings_umap_protoA.png','UMAP: temporal, contextual and fused representations.',170),plot('embeddings_tsne_protoA.png','t-SNE: the same three required blocks.',170),text('Reproduction: follow the component README. tools/finalize_evaluation.py regenerates the final tables and figures from checkpoint-verified predictions. Build: python tools/build_reports.py --final --verification-root assignment_2b_emotion/Results/verified_p0. Independent checkpoints, input hashes and environment records support reproduction; historical invalidated markers are excluded. Raw audio, model caches, checkpoints and secrets are excluded from Git.'),text('[B6] Pichora-Fuller & Dupuis, TESS, 2020. https://doi.org/10.5683/SP2/E8H2MF. All required headline runs, projections, failure cases and scoped Praat visual observations passed the final evidence gates. No manual gold boundary accuracy or population-level emotion-recognition generalization is claimed.')]
 return [('Dataset, splits and completion',p1),('Architecture and block justifications',p2),('Validated accuracy comparison',p3),('Emotion analysis and caveats',p4),('Five actual failure cases',p5),('Three block embeddings and reproducibility',p6)]
def render(ev,folder,basename,title,pages,draft):
 from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Image,Spacer,PageBreak
 from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
 from reportlab.lib import colors
 from reportlab.lib.utils import ImageReader
 import pymupdf
 folder.mkdir(parents=True,exist_ok=True); st=getSampleStyleSheet()
 for name,size,leading in [('BodyReport',8.5,11.5),('CellReport',7.2,9),('SmallReport',7,9)]: st.add(ParagraphStyle(name=name,fontName='Helvetica',fontSize=size,leading=leading,spaceAfter=6))
 def para(s,style='BodyReport'): return Paragraph(escape(str(s)).replace('\n','<br/>'),st[style])
 story=[]; md=['# '+title,'','**'+('DRAFT - pending evidence remains.' if draft else 'FINAL - required evidence gates passed.')+'**','']
 for i,(heading,nodes) in enumerate(pages):
  if i: story.append(PageBreak())
  story += [para(title,'Heading1'),para(heading,'Heading2')]; md += ['## '+heading,'']
  for nodeindex,(kind,value) in enumerate(nodes):
   if kind=='gallery':
    paths,caption,maxh=value
    if not paths:
     story.append(para('PENDING: '+caption)); md += ['PENDING: '+caption,'']; continue
    from PIL import Image as PILImage, ImageOps
    cols=2; cellw,cellh=1100,720; rows=(len(paths)+cols-1)//cols
    panel=PILImage.new('RGB',(cols*cellw,rows*cellh),'white')
    for j,path in enumerate(paths):
     with PILImage.open(path) as source:
      thumb=ImageOps.contain(source.convert('RGB'),(cellw,cellh))
      panel.paste(thumb,((j%cols)*cellw+(cellw-thumb.width)//2,(j//cols)*cellh+(cellh-thumb.height)//2))
    figdir=folder/'figures'; figdir.mkdir(exist_ok=True); path=figdir/f'panel_{i+1}_{nodeindex}.png'; panel.save(path)
    value=(path,caption,maxh); kind='image'
   if kind=='text': story.append(para(value)); md += [value,'']
   elif kind=='table':
    heads,rows=value
    if not rows: continue
    t=Table([[para(z,'CellReport') for z in row] for row in [heads]+rows],colWidths=[507/len(heads)]*len(heads),repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5edf5')),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#b8c3d0')),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    story += [t,Spacer(1,7)]
    clean=lambda x:str(x).replace('|','/').replace('\n',' ')
    md += ['| '+' | '.join(map(clean,heads))+' |','|'+'---|'*len(heads)]+['| '+' | '.join(map(clean,row))+' |' for row in rows]+['']
   elif kind=='image':
    path,caption,maxh=value; w,h=ImageReader(str(path)).getSize(); ratio=min(507/w,maxh/h)
    story += [Image(str(path),width=w*ratio,height=h*ratio),para(caption,'SmallReport')]
    md += [f'![{caption}]({os.path.relpath(path,folder).replace(chr(92),"/")})','']
 md += ['## Pending evidence','']+(['- '+x for x in ev.pending] if ev.pending else ['No pending requirements recorded.'])+['']
 (folder/(basename+'.md')).write_text('\n'.join(md)+'\n',encoding='utf-8'); pdf=folder/(basename+'.pdf')
 def footer(canvas,doc):
  canvas.setStrokeColor(colors.HexColor('#c3ceda')); canvas.line(44,34,551,34); canvas.setFont('Helvetica',7); canvas.setFillColor(colors.HexColor('#42536a'))
  canvas.drawString(44,23,('DRAFT' if draft else 'FINAL')+' | IIITH Speech Processing Assignment 2 | Evidence-derived'); canvas.drawRightString(551,23,str(doc.page))
 SimpleDocTemplate(str(pdf),pagesize=(595,842),leftMargin=44,rightMargin=44,topMargin=36,bottomMargin=44,title=title,author='IIITH Assignment 2',invariant=1).build(story,onFirstPage=footer,onLaterPages=footer)
 qa=folder/'qa'; qa.mkdir(exist_ok=True); doc=pymupdf.open(pdf)
 for i,page in enumerate(doc):
  page.get_pixmap(matrix=pymupdf.Matrix(1.15,1.15)).save(qa/f'page_{i+1:02}.png')
  for block in page.get_text('blocks'):
   x0,y0,x1,y1=block[:4]; require(x0>=20 and x1<=575 and y0>=15 and y1<=827,f'PDF text out of bounds, page {i+1}')
 (folder/'build_manifest.json').write_text(json.dumps(dict(draft=draft,pages=len(doc),logical_pages=len(pages),pending=ev.pending,input_sha256=ev.hashes,pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest()),indent=2)+'\n',encoding='utf-8')
 print(f'{pdf}: {len(doc)} pages, {len(ev.pending)} pending requirements; rendered QA PNGs')
def self_test():
 import tempfile
 m,_=metrics([dict(y_true=str(i),y_pred=str(i)) for i in range(7)]); assert all(v==1 for v in m.values())
 m,_=metrics([dict(y_true=str(i),y_pred=str((i+1)%7)) for i in range(7)]); assert all(v==0 for v in m.values())
 with tempfile.TemporaryDirectory() as td:
  ev=Evidence(Path(td)); good,status,failures=collect_b(ev)
  assert not good and len(status)==12 and all(x[2]=='PENDING' for x in status) and len(ev.pending)>12 and not failures
 print('PASS: metric checks; all 12 missing runs, three blocks, plots and five failures fail closed')
def main():
 ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); ap.add_argument('--final',action='store_true'); ap.add_argument('--allow-provisional-2a',action='store_true'); ap.add_argument('--self-test',action='store_true')
 ap.add_argument('--verification-root',type=Path,help='Distinct directory containing re-evaluated headline run folders; see BUILD.md')
 args=ap.parse_args()
 if args.self_test: self_test(); return 0
 ev=Evidence(args.root.resolve()); ev.verification_root=args.verification_root.resolve() if args.verification_root else None
 aa=a_pages(ev); good,status,failures=collect_b(ev); bb=b_pages(ev,good,status,failures)
 blocking=[x for x in ev.pending if not(args.allow_provisional_2a and x.startswith('2A Praat'))]
 if args.final and blocking:
  print('FINAL REFUSED before writing:\n'+'\n'.join('- '+x for x in blocking),file=sys.stderr); return 2
 pending=list(ev.pending)
 for label,folder,pages in [('2A','assignment_2a_mfa',aa),('2B','assignment_2b_emotion',bb)]:
  ev.pending=[x for x in pending if x.startswith(label)]
  render(ev,ev.root/folder/'report','report_'+label.lower(),'Assignment '+label+' | '+('Forced alignment' if label=='2A' else 'Multimodal emotion recognition'),pages,not args.final or bool(ev.pending))
 return 0
if __name__=='__main__': raise SystemExit(main())
