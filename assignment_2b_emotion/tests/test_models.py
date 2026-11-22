"""Shape and sanity unit tests for SpeechEmotionModel and temporal/pooling blocks."""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mer.models.speech import SpeechEmotionModel


def test_speech_model_shapes():
    for arch in ("gru", "lstm", "transformer", "cnn", "none"):
        for pool in ("asp", "mean"):
            m = SpeechEmotionModel(num_layers=13, in_dim=768, proj_dim=64, temporal_arch=arch,
                                   temporal_hidden=32, pooling=pool, head_hidden=32)
            x = torch.randn(4, 13, 50, 768)
            mask = torch.ones(4, 50, dtype=torch.bool)
            mask[0, 30:] = False
            out = m(x, mask)
            assert out["logits"].shape == (4, 7)
            assert out["temporal_emb"].ndim == 2 and out["temporal_emb"].shape[0] == 4
            assert torch.isclose(out["layer_weights"].sum(), torch.tensor(1.0), atol=1e-4)
