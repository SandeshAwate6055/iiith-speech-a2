import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('audit_lexicon', Path(__file__).resolve().parents[1] / 'scripts/audit_lexicon.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_missing_dictionary_is_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        module.read_lexicon(tmp_path / 'missing.dict')


def test_probabilities_and_phone_validation(tmp_path):
    p = tmp_path / 'test.dict'
    p.write_text('hello 0.5 0.2 1.0 1.0 HH AH0 L OW1\nwrong HH AH4\nempty\n', encoding='utf-8')
    entries, errors = module.read_lexicon(p)
    assert entries['hello'] == {('HH', 'AH0', 'L', 'OW1')}
    assert len(errors) == 2


def test_coverage_detects_oov_and_validates_complete_corpus(tmp_path):
    corpus = tmp_path / 'corpus'
    corpus.mkdir()
    (corpus / 'sample.lab').write_text('HELLO unknown hello', encoding='utf-8')
    d = tmp_path / 'test.dict'
    d.write_text('hello HH AH0 L OW1\n', encoding='utf-8')
    result = module.audit(d, corpus)
    assert result['oovs'] == {'unknown': ['sample']}
    assert not result['passed']
    d.write_text('hello HH AH0 L OW1\nunknown AH0 N N OW1 N\n', encoding='utf-8')
    assert module.audit(d, corpus)['passed']
