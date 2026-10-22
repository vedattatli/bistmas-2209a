from bistmas.shared.config import load_settings


def test_settings_load_and_paths_exist(settings):
    assert settings.paths.raw.exists()
    assert settings.paths.store.exists()
    assert settings.jury.model == "qwen3.5:9b"
    assert set(settings.jury.sampling_options()) == {"num_ctx", "num_predict", "temperature", "seed", "think"}
    assert settings.backtest.headline_top_n in settings.backtest.top_n


def test_reload_is_deterministic():
    a = load_settings().model_dump(exclude={"raw_config"})
    b = load_settings().model_dump(exclude={"raw_config"})
    assert a == b
