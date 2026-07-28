import os
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

from src.data_collector import download_stock_data, save_raw_data
from src.preprocessor import Preprocessor
from src.lstm_model import create_lstm_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

SYMBOL = "NVDA"
START_DATE = "2020-01-01"
END_DATE = "2026-07-24"
SEQ_LENGTH = 60
BATCH_SIZE = 32
EPOCHS = 100
MODEL_DIR = "models"


def main():
    print(f"Downloading {SYMBOL} data from {START_DATE} to {END_DATE}...")
    df = download_stock_data(SYMBOL, START_DATE, END_DATE)
    save_raw_data(df, SYMBOL)
    print(f"  Got {len(df)} rows — {df.index[0].date()} to {df.index[-1].date()}")

    print("Preprocessing...")
    pp = Preprocessor(seq_length=SEQ_LENGTH)
    pp.fit(df)
    scaled = pp.transform(df)
    X, y = pp.create_sequences(scaled)
    X_train, y_train, X_val, y_val, X_test, y_test = pp.split_data(X, y)
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print("Building LSTM model...")
    model = create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)

    print("Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )

    print(f"Saving model and scaler to '{MODEL_DIR}/'...")
    model.save(f"{MODEL_DIR}/lstm_model.keras")
    pp.save(f"{MODEL_DIR}/preprocessor.pkl")

    print("Done.")

    return model, pp, X_test, y_test


if __name__ == "__main__":
    main()
