from pathlib import Path
import sys
import time
import numpy as np

# Dodajemy src do sciezki, zeby importy dzialaly z poziomu scripts/
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from data_loader import load_and_preprocess_data, split_dataset, scale_features


def main():
    data_path = Path(__file__).resolve().parent.parent / "data" / "csgo_round_snapshots.csv"
    if not data_path.exists():
        print(f"Brak pliku z danymi w {data_path}")
        return

    print("--- Test ladowania i przetwarzania danych CS:GO ---")

    # Wczytanie
    t0 = time.time()
    X, y = load_and_preprocess_data(data_path, dtype=np.float32)
    t_load = time.time() - t0

    print(f"Wczytano {X.shape[0]} wierszy i {X.shape[1]} kolumn w {t_load:.2f}s")
    print(f"Rozmiar w RAM: X = {X.nbytes / 1e6:.1f} MB, y = {y.nbytes / 1e6:.2f} MB")
    print(f"C_CONTIGUOUS: X={X.flags['C_CONTIGUOUS']}, y={y.flags['C_CONTIGUOUS']}")
    print(f"Balans klas: CT={np.mean(y == 1)*100:.1f}%, T={np.mean(y == 0)*100:.1f}%")

    # 2. Podzial train/test
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2, seed=42)
    print(f"\nPodzial train/test: train={X_train.shape}, test={X_test.shape}")
    print(f"Uklad w pamieci po podziale: train={X_train.flags['C_CONTIGUOUS']}, test={X_test.flags['C_CONTIGUOUS']}")

    # Skalowanie
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    print(f"\nSkalowanie zakonczone.")
    print(f"Srednia w trainie: ~{np.mean(X_train_scaled):.4f}, std: ~{np.std(X_train_scaled):.4f}")
    print(f"Typ po skalowaniu: {X_train_scaled.dtype}")

    # Sprawdzenie wycinania batcha
    batch = X_train_scaled[:64]
    print(f"\nBatch 64 probek: shape={batch.shape}, C_CONTIGUOUS={batch.flags['C_CONTIGUOUS']}")
    print("Wszystko dziala poprawnie!")


if __name__ == "__main__":
    main()
