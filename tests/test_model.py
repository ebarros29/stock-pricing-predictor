import pytest
from src.lstm_model import create_lstm_model


def test_model_creates_and_compiles():
    model = create_lstm_model(input_shape=(60, 5))
    assert model is not None

    # Check architecture layers in order
    layer_names = [l.name for l in model.layers]
    assert "lstm" in layer_names[0]
    assert "lstm_1" in layer_names[2]  # after dropout[0]
    assert "dense" in layer_names[-2]
    assert "dense_1" in layer_names[-1]


def test_model_output_shape():
    model = create_lstm_model(input_shape=(60, 5))
    # input: (batch, 60 timesteps, 5 features) → output: (batch, 1)
    assert model.output_shape == (None, 1)


def test_model_with_custom_units():
    model = create_lstm_model(input_shape=(60, 3), lstm_units=[64, 32, 16])
    lstm_layers = [l for l in model.layers if "lstm" in l.name]
    assert len(lstm_layers) == 3
    assert lstm_layers[0].units == 64
    assert lstm_layers[1].units == 32
    assert lstm_layers[2].units == 16


def test_model_compiles_with_adam_and_mse():
    model = create_lstm_model(input_shape=(60, 5))
    config = model.optimizer.get_config()
    assert "adam" in config["name"].lower()


def test_different_input_features():
    for n_features in [1, 3, 10]:
        model = create_lstm_model(input_shape=(30, n_features))
        assert model.input_shape == (None, 30, n_features)
