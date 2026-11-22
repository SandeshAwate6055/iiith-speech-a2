"""Build a corpus-scoped ARPAbet lexicon with explicit pronunciation provenance."""
import csv
import json
from pathlib import Path
from audit_lexicon import read_lexicon, audit


def main():
    root=Path(__file__).resolve().parents[1]
    words=set(' '.join(p.read_text(encoding='utf-8') for p in (root/'corpus').rglob('*.lab')).lower().split())
    inputs={'base':root/'work/models/hf_arpa.dict','g2p':root/'dictionaries/g2p_oov.dict','manual':root/'dictionaries/manual_overrides.dict'}
    layers={}
    for name,path in inputs.items():
        entries,errors=read_lexicon(path)
        if errors:raise ValueError(f'{name}: invalid ARPAbet entries: {errors[:3]}')
        layers[name]=entries
    model=json.loads((root/'work/models/hf_source.json').read_text())
    inventory=set(model['acoustic_metadata']['phones'])
    merged,rows={},[]
    for word in sorted(words):
        source=next((s for s in ['manual','g2p','base'] if word in layers[s]),None)
        if source is None:raise ValueError(f'Uncovered word: {word}')
        merged[word]=layers[source][word]
        for phones in sorted(merged[word]):
            if not set(phones)<=inventory:raise ValueError(f'Phones absent in acoustic model: {word} {phones}')
            rows.append(dict(word=word,pronunciation=' '.join(phones),source=source,review_status='candidate; acoustic and listening review required'))
    out=root/'dictionaries/custom_lexicon.dict'
    out.write_text(''.join(f"{r['word']}\t{r['pronunciation']}\n" for r in rows),encoding='utf-8')
    evaluation=root/'outputs/evaluation';evaluation.mkdir(parents=True,exist_ok=True)
    with (evaluation/'lexicon_pronunciation_sources.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    report=audit(out,root/'corpus')
    report.update(base_oovs=sorted(words-layers['base'].keys()),acoustic_phones_compatible=True,model_revision=model['revision'],license_note='Base pronunciations/model: MontrealCorpusTools english_us_arpa, CC BY 4.0; corpus-scoped derivatives with G2P and manual candidates. No claim of human gold verification.')
    (evaluation/'lexicon_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    assert report['passed']
    print(f"Covered {len(words)} types, {len(rows)} pronunciations, zero OOV, acoustic inventory compatible")

if __name__=='__main__':main()
