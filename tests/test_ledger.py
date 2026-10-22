"""Experiment ledger: append-only lines, path-independent config hash, trial counting."""

from pathlib import Path

from bistmas.shared.config import PathsConfig, Settings
from bistmas.shared.ledger import config_hash, read_ledger, record_run, trial_count


def _settings(tmp_path: Path) -> Settings:
    return Settings(root=tmp_path, paths=PathsConfig().resolve(tmp_path))


def test_record_and_trial_count(tmp_path: Path):
    s = _settings(tmp_path)
    assert trial_count(s) == 0
    record_run(s, "deterministic", "det-a", {"det_top3": {"cagr_tl": 0.1}, "bench_ew_pit": {"cagr_tl": 0.05}})
    record_run(s, "deterministic", "det-b", {"det_top3": {"cagr_tl": 0.2}, "det_top5": {"cagr_tl": 0.2}})
    record_run(s, "jury", "prereg-v1", None, {"records": 10})
    lines = read_ledger(s)
    assert [e["kind"] for e in lines] == ["deterministic", "deterministic", "jury"]
    assert lines[0]["config_hash"] == lines[1]["config_hash"] and lines[2]["records"] == 10
    # same config + same label counts once; the benchmark never counts; jury lines are not portfolio trials
    assert trial_count(s) == 2
    record_run(s, "hybrid", "hyb", {"hybrid_top3": {}, "det_top3": {}})
    assert trial_count(s) == 3  # det_top3 is shared with the deterministic run; only hybrid_top3 is new
    # the run being recorded counts exactly once, also when it is already in the ledger
    assert trial_count(s, extra_labels=["det_top3", "hybrid_top3", "bench_ew_pit"]) == 3
    assert trial_count(s, extra_labels=["det_top10", "hybrid_top10"]) == 4  # hybrid_top10 == det_top10 (Top-N screen)
    record_run(s, "hybrid", "hyb2", {"hybrid_top10": {}})
    assert trial_count(s) == 4


def test_config_hash_ignores_machine_paths(tmp_path: Path):
    a = _settings(tmp_path / "one")
    b = _settings(tmp_path / "two")
    assert config_hash(a) == config_hash(b)
    c = _settings(tmp_path / "three")
    c.jury.theta = 0.9
    assert config_hash(c) != config_hash(a)


def test_store_vintages_record_curated_config_files(tmp_path: Path):
    from bistmas.shared.ledger import store_vintages

    v = store_vintages(_settings(tmp_path))
    assert v["config_ticker_aliases"] and v["config_universe_overrides"] and "universe_membership" in v
