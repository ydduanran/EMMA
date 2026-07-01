import pytest

from emma_3dgenome.config import EmmaConfig, get_preset_config


def test_default_config_valid():
    cfg = get_preset_config("default")
    cfg.validate()
    assert cfg.max_imfs == 5
    assert len(cfg.imf_weights) == cfg.max_imfs


def test_wrong_imf_weight_length_raises():
    cfg = EmmaConfig(max_imfs=3, imf_weights=(1.0, 1.0))
    with pytest.raises(ValueError, match="len\\(imf_weights\\) must equal max_imfs"):
        cfg.validate()


def test_fast_preset_loads():
    cfg = get_preset_config("fast")
    assert cfg.epochs == 5
    assert cfg.max_diag == 300
