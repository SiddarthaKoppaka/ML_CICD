from training_pipeline.config_loader import load_config


def test_config_loading():
    config = load_config("training_pipeline/config.yaml")
    assert "data" in config
    assert "training" in config
    assert "output" in config
    assert "feature_path" in config["data"]
