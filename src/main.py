import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
from numpy.typing import NDArray

# Upewniamy sie, ze katalog src jest w sys.path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import load_and_preprocess_data, split_dataset, scale_features
from sequential_sgd import add_intercept, generate_synthetic_dataset, train_sequential_sgd
from numba_sgd import train_numba_sgd


def save_model(
    filepath: Path | str,
    weights: NDArray[Any],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """
    Zapisuje wyuczone wagi modelu do pliku .npy lub .npz (z metadanymi).
    Automatycznie tworzy brakujace katalogi nadrzedne.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".npz":
        meta_dict = metadata or {}
        np.savez(path, weights=weights, **meta_dict)
    else:
        # Domyslnie format .npy
        if path.suffix != ".npy":
            path = path.with_suffix(".npy")
        np.save(path, weights)

    return path


def load_model(filepath: Path | str) -> tuple[NDArray[Any], dict[str, Any] | None]:
    """
    Wczytuje wagi modelu z pliku .npy lub .npz.
    Zwraca krotke (weights, metadata).
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku modelu: {path}")

    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            weights = data["weights"]
            metadata = {k: data[k].item() if data[k].ndim == 0 else data[k] for k in data.files if k != "weights"}
            return weights, metadata
    else:
        weights = np.load(path)
        return weights, None


def _json_serializable(val: Any) -> Any:
    """Konwerter typow NumPy i standardowych na formaty JSON."""
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (float, np.floating)):
        return float(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    return str(val)


def save_history(filepath: Path | str, history: dict[str, Any]) -> Path:
    """
    Zapisuje slownik z historia metryk (Loss, Accuracy, Epoch Times) do pliku JSON.
    Automatycznie tworzy brakujace katalogi nadrzedne.
    """
    path = Path(filepath)
    if path.suffix != ".json":
        path = path.with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)

    # Konwertujemy wszystkie wartosci NumPy na typy serializowalne w JSON
    serializable_history: dict[str, Any] = {}
    for k, v in history.items():
        if isinstance(v, list):
            serializable_history[k] = [_json_serializable(item) for item in v]
        else:
            serializable_history[k] = _json_serializable(v)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable_history, f, indent=2, ensure_ascii=False)

    return path


def load_history(filepath: Path | str) -> dict[str, Any]:
    """
    Wczytuje historie metryk uczenia z pliku JSON.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku historii: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_time_to_quality(
    history: dict[str, Any],
    target_accuracy: float,
) -> tuple[float | None, int | None]:
    """
    Wyznacza metryke Time-to-Quality:
    - czas od startu do pierwszej epoki, w ktorej osiagnieto zadany prog dokladnosci target_accuracy,
    - numer tej epoki (1-indexed).
    Jesli prog nie zostal osiagniety, zwraca (None, None).
    """
    accuracies = history.get("accuracy", [])
    epoch_times = history.get("epoch_times", [])

    for i, acc in enumerate(accuracies):
        if acc >= target_accuracy:
            epoch_num = i + 1
            if epoch_times and len(epoch_times) >= epoch_num:
                time_to_q = float(sum(epoch_times[:epoch_num]))
            else:
                time_to_q = 0.0
            return time_to_q, epoch_num

    return None, None


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """
    Definiuje i parsuje argumenty wiersza polecen zgodnie ze specyfikacja tematu D09.
    """
    parser = argparse.ArgumentParser(
        description="Regresja logistyczna z redukcja gradientow (PRiR D09) - Uczenie Mini-Batch SGD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["seq", "numba", "mpi", "cuda"],
        default="seq",
        help="Wariant obliczeniowy uczenia modelu",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Wspolczynnik uczenia (learning rate)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Wielkosc partii (mini-batch size)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Liczba epok uczenia",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Ziarno generatora liczb losowych",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/csgo_round_snapshots.csv",
        help="Sciezka do pliku z danymi CSV",
    )
    parser.add_argument(
        "--target-acc",
        type=float,
        default=0.75,
        help="Opcjonalny docelowy prog dokladnosci do pomiaru Time-to-Quality (np. 0.75)",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        help="Sciezka do zapisu wag modelu (format .npy lub .npz)",
    )
    parser.add_argument(
        "--save-history",
        type=str,
        default=None,
        help="Sciezka do zapisu historii metryk (format .json)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Udzial zbioru testowego w podziale",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Czestotliwosc synchronizacji K dla wariantu MPI Local SGD",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Wyciszenie szczegolowych logow epok",
    )

    return parser.parse_args(args)


def run_training(args: argparse.Namespace) -> tuple[NDArray[Any], dict[str, Any]]:
    """
    Przygotowuje dane, trenuje model w zadanym trybie i zwraca (weights, history).
    """
    is_main_process = True
    comm = None

    # Obsluga srodowiska MPI
    if args.mode == "mpi":
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            is_main_process = (comm.Get_rank() == 0)
        except Exception:
            comm = None
            is_main_process = True

    # 1. Wczytanie i preprocessing danych
    data_path = Path(args.data)
    if not data_path.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        data_path = repo_root / args.data

    if data_path.is_file():
        if is_main_process and not args.quiet:
            print(f"[Data] Wczytywanie danych z: {data_path}")
        X, y = load_and_preprocess_data(data_path, dtype=np.float32)
    else:
        if is_main_process and not args.quiet:
            print(f"[Data] Brak pliku {data_path} - generowanie syntetycznego zbioru (4000 probek, 12 cech).")
        X, y = generate_synthetic_dataset(n_samples=4000, n_features=12, seed=args.seed, dtype=np.float32)

    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=args.test_size, seed=args.seed)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    X_train_final = add_intercept(X_train_scaled)
    X_test_final = add_intercept(X_test_scaled)

    if is_main_process and not args.quiet:
        print(
            f"[Start] Tryb: {args.mode.upper()} | Train: {X_train_final.shape} | Test: {X_test_final.shape} | "
            f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr} | Seed: {args.seed}"
        )

    verbose = (not args.quiet) and is_main_process

    # 2. Uruchomienie wybranego wariantu obliczeniowego
    if args.mode == "seq":
        weights, history = train_sequential_sgd(
            X_train_final,
            y_train,
            X_val=X_test_final,
            y_val=y_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed,
            verbose=verbose,
        )

    elif args.mode == "numba":
        weights, history = train_numba_sgd(
            X_train_final,
            y_train,
            X_val=X_test_final,
            y_val=y_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed,
            verbose=verbose,
        )

    elif args.mode == "mpi":
        if args.k and args.k > 1:
            from model_mpi import train_local_sgd_mpi
            weights, history = train_local_sgd_mpi(
                X_train_final,
                y_train,
                X_val=X_test_final,
                y_val=y_test,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                sync_freq_k=args.k,
                seed=args.seed,
                comm=comm,
                verbose=verbose,
            )
        else:
            from model_mpi import train_mpi_sgd
            weights, history = train_mpi_sgd(
                X_train_final,
                y_train,
                X_val=X_test_final,
                y_val=y_test,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                seed=args.seed,
                comm=comm,
                verbose=verbose,
            )

    elif args.mode == "cuda":
        from cuda_sgd import train_cuda_sgd
        weights, history = train_cuda_sgd(
            X_train_final,
            y_train,
            X_val=X_test_final,
            y_val=y_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Nieznany tryb: {args.mode}")

    # 3. Wyznaczenie metryki Time-to-Quality
    if is_main_process and args.target_acc is not None:
        time_to_q, ep_to_q = calculate_time_to_quality(history, args.target_acc)
        history["target_accuracy"] = args.target_acc
        history["time_to_target_accuracy"] = time_to_q
        history["epoch_to_target_accuracy"] = ep_to_q

        if verbose:
            print("-" * 64)
            if time_to_q is not None and ep_to_q is not None:
                print(
                    f"[Time-to-Quality] Prog dokladnosci {args.target_acc * 100:.2f}% "
                    f"osiagnieto w epoce {ep_to_q:d} w czasie {time_to_q:.4f}s"
                )
            else:
                max_acc = max(history.get("accuracy", [0.0]))
                print(
                    f"[Time-to-Quality] Prog dokladnosci {args.target_acc * 100:.2f}% "
                    f"NIE zostal osiagniety (maksymalna dokladnosc: {max_acc * 100:.2f}%)"
                )

    # 4. Zapis modelu i historii
    if is_main_process:
        if args.save_model:
            saved_m_path = save_model(
                args.save_model,
                weights,
                metadata={
                    "mode": args.mode,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "seed": args.seed,
                },
            )
            if verbose:
                print(f"[Save] Wagi modelu zapisano do: {saved_m_path}")

        if args.save_history:
            saved_h_path = save_history(args.save_history, history)
            if verbose:
                print(f"[Save] Historie uczenia zapisano do: {saved_h_path}")

    return weights, history


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
