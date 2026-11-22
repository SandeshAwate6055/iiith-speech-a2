"""Strict corpus coverage and ARPAbet checks, independent of active MFA downloads.
Usage: python scripts/audit_lexicon.py --dictionary PATH --corpus corpus --out outputs/evaluation/lexicon_audit.json
Optional MFA probability columns are parsed, not interpreted as phones.
This is lexical validation, not a substitute for mfa validate or listening.
"""
import argparse
import json
import re
from pathlib import Path

VOWELS = set('AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW'.split())
CONSONANTS = set('B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH'.split())
SPECIAL = {'sil', 'sp', 'spn', '<eps>'}


def is_phone(phone):
    if phone in SPECIAL or phone in CONSONANTS:
        return True
    match = re.fullmatch(r'([A-Z]+)([012]?)', phone)
    return bool(match and match[1] in VOWELS)


def read_lexicon(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    entries, errors = {}, []
    for n, raw in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        parts = raw.strip().split()
        if not parts or parts[0].startswith('#'):
            continue
        word, phones = parts[0].lower(), parts[1:]
        while phones:
            try:
                float(phones[0])
            except ValueError:
                break
            phones = phones[1:]
        if not phones:
            errors.append({'line': n, 'word': word, 'reason': 'empty pronunciation'})
            continue
        invalid = [p for p in phones if not is_phone(p)]
        if invalid:
            errors.append({'line': n, 'word': word, 'reason': 'invalid ARPAbet phones', 'phones': invalid})
            continue
        entries.setdefault(word, set()).add(tuple(phones))
    return entries, errors


def audit(dictionary, corpus):
    entries, errors = read_lexicon(dictionary)
    labs = sorted(Path(corpus).rglob('*.lab'))
    if not labs:
        raise ValueError(f'No .lab files in {corpus}')
    coverage, oovs = [], {}
    for lab in labs:
        words = lab.read_text(encoding='utf-8-sig').lower().split()
        missing = sorted(set(words) - set(entries))
        for w in missing:
            oovs.setdefault(w, []).append(lab.stem)
        coverage.append({'file': lab.stem, 'tokens': len(words), 'oov_types': missing})
    return dict(dictionary=str(dictionary), corpus=str(corpus), files=len(labs), dictionary_words=len(entries), invalid_entries=errors, oov_types=len(oovs), oovs=oovs, coverage=coverage, passed=not errors and not oovs, limitation='ARPAbet syntax and coverage only; pronunciation accuracy and acoustic compatibility require MFA validation and listening.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dictionary', required=True)
    p.add_argument('--corpus', default='corpus')
    p.add_argument('--out', default='outputs/evaluation/lexicon_audit.json')
    a = p.parse_args()
    result = audit(a.dictionary, a.corpus)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"files={result['files']} OOV types={result['oov_types']} invalid entries={len(result['invalid_entries'])}")
    raise SystemExit(0 if result['passed'] else 1)

if __name__ == '__main__':
    main()
