import json
import os
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dia.config import (
    DataConfig,
    DecoderConfig,
    DiaConfig,
    EncoderConfig,
    ModelConfig,
    TrainingConfig,
)


def minimal_config() -> DiaConfig:
    return DiaConfig(
        model=ModelConfig(
            encoder=EncoderConfig(
                n_layer=1,
                n_embd=8,
                n_hidden=8,
                n_head=1,
                head_dim=8,
            ),
            decoder=DecoderConfig(
                n_layer=1,
                n_embd=8,
                n_hidden=8,
                gqa_query_heads=1,
                kv_heads=1,
                gqa_head_dim=8,
                cross_query_heads=1,
                cross_head_dim=8,
            ),
        ),
        training=TrainingConfig(),
        data=DataConfig(text_length=128, audio_length=128),
    )


def test_save_without_directory(tmp_path, monkeypatch):
    cfg = minimal_config()
    monkeypatch.chdir(tmp_path)
    path = "tmp_config.json"
    cfg.save(path)
    assert os.path.isfile(path)
    with open(path, "r") as f:
        json.load(f)
    os.remove(path)


def test_save_with_directory(tmp_path):
    cfg = minimal_config()
    path = tmp_path / "dir1" / "config.json"
    cfg.save(str(path))
    assert path.is_file()
    with open(path, "r") as f:
        json.load(f)


def test_save_invalid_extension(tmp_path):
    cfg = minimal_config()
    invalid_path = tmp_path / "config.txt"
    with pytest.raises(ValueError):
        cfg.save(str(invalid_path))
