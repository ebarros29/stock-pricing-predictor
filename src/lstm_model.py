from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def create_lstm_model(input_shape: tuple, lstm_units: list[int] = None, dropout: float = 0.2):
    lstm_units = lstm_units or [50, 50]
    model = Sequential()
    for i, units in enumerate(lstm_units):
        return_seq = i < len(lstm_units) - 1
        if i == 0:
            model.add(LSTM(units, return_sequences=return_seq, input_shape=input_shape))
        else:
            model.add(LSTM(units, return_sequences=return_seq))
        model.add(Dropout(dropout))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model
