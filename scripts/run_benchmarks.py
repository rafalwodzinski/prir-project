from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
import numpy as np

# Dodajemy katalog src do sys.path, zeby moc importowac moduly projektu
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from numba import get_num_threads, set_num_threads
from data_loader import load_and_preprocess_data, split_dataset, scale_features
from sequential_sgd import add_intercept, compute_accuracy, generate_synthetic_dataset, predict_proba, train_sequential_sgd
from numba_sgd import compute_gradient_numba, train_numba_sgd
from model_mpi import split_batch_for_rank, train_local_sgd_mpi, MPI
from math_utils import sigmoid


def get_hardware_info() -> dict[str, Any]:
    """
    Pobiera podstawowe informacje o srodowisku sprzetowym do legendy wykresow.
    """
    return {
        "processor": platform.processor() or "CPU x86_64",
        "cpu_count_logical": os.cpu_count() or 1,
        "python_version": platform.python_version(),
        "system": platform.system(),
        "release": platform.release(),
    }


def measure_median_time(
    benchmark_fn: Any,
    warmup_runs: int = 1,
    measurement_runs: int = 5,
) -> tuple[float, list[float]]:
    """
    Wykonuje pomiar czasu z odrzuceniem przebiegow rozgrzewkowych (warm-up).
    Zbiera zadana liczbe probek i zwraca mediane oraz pelna liste czasow.
    """
    # 1. Rozgrzewka (warm-up) - kompilacja Numba JIT, ladowanie do pamieci cache CPU
    for _ in range(warmup_runs):
        benchmark_fn()

    # 2. Wlasciwe pomiary
    times: list[float] = []
    for _ in range(measurement_runs):
        t0 = time.perf_counter()
        benchmark_fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    median_t = float(np.median(times))
    return median_t, times


def benchmark_openmp_numba(
    X: np.ndarray,
    y: np.ndarray,
    thread_counts: list[int] | None = None,
    epochs: int = 5,
    batch_size: int = 256,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Bada skalowalnosc wielowatkowej implementacji Numba (odpowiednik OpenMP na CPU).
    Mierzy czasy dla roznych liczb watkow z uzyciem set_num_threads.
    """
    max_cores = os.cpu_count() or 4
    if thread_counts is None:
        # Dobieramy sensowne liczby watkow np. 1, 2, 4, 8 (nie przekraczajac dostepnych)
        candidates = [1, 2, 4, 8, 16]
        thread_counts = [t for t in candidates if t <= max_cores]
        if 1 not in thread_counts:
            thread_counts.insert(0, 1)

    print(f"\n--- Benchmark OpenMP / Numba (watki: {thread_counts}) ---")
    results_by_threads: dict[int, dict[str, Any]] = {}

    for t in thread_counts:
        set_num_threads(t)

        def run_once() -> None:
            train_numba_sgd(
                X,
                y,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=0.01,
                seed=seed,
                verbose=False,
            )

        med_time, samples = measure_median_time(run_once, warmup_runs=1, measurement_runs=5)
        print(f"Watki p={t:2d} | Mediana: {med_time:.4f}s | Probki: {[round(s, 4) for s in samples]}")

        results_by_threads[t] = {
            "threads": t,
            "median_time": med_time,
            "samples": samples,
            "min_time": float(np.min(samples)),
            "max_time": float(np.max(samples)),
        }

    # Obliczamy przyspieszenie S(p) = T_1 / T_p oraz efektywnosc E(p) = S(p) / p
    t1 = results_by_threads[thread_counts[0]]["median_time"]
    for t in thread_counts:
        tp = results_by_threads[t]["median_time"]
        speedup = t1 / tp if tp > 0 else 1.0
        efficiency = speedup / t
        results_by_threads[t]["speedup"] = float(speedup)
        results_by_threads[t]["efficiency"] = float(efficiency)

    return {
        "thread_counts": thread_counts,
        "results": {str(k): v for k, v in results_by_threads.items()},
        "baseline_time_t1": t1,
    }


def simulate_local_sgd_multiprocess(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_workers: int,
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 0.01,
    sync_freq_k: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Symulacja rozproszonego Local SGD z K-krokowa synchronizacja wag dla n_workers procesow.
    Pozwala na wiarygodny pomiar narzutu podzialu danych i synchronizacji Allreduce
    w dowolnym srodowisku (w tym Windows bez natywnego MS-MPI).
    """
    rng = np.random.default_rng(seed)
    n_samples, n_features = X_train.shape

    # Kazdy proces ma swoje lokalne wagi
    worker_weights = [np.zeros(n_features, dtype=X_train.dtype) for _ in range(n_workers)]
    step_counter = 0
    sync_count = 0

    for epoch in range(1, epochs + 1):
        indices = rng.permutation(n_samples)
        X_shuffled = np.ascontiguousarray(X_train[indices])
        y_shuffled = np.ascontiguousarray(y_train[indices])

        for start_idx in range(0, n_samples, batch_size):
            end_idx = start_idx + batch_size
            X_global_batch = X_shuffled[start_idx:end_idx]
            y_global_batch = y_shuffled[start_idx:end_idx]

            # Faza lokalna na kazdym wezle (brak komunikacji sieciowej)
            for rank in range(n_workers):
                X_loc, y_loc = split_batch_for_rank(X_global_batch, y_global_batch, rank, n_workers)
                if X_loc.shape[0] > 0:
                    y_pred = sigmoid(X_loc @ worker_weights[rank])
                    grad_loc = (X_loc.T @ (y_pred - y_loc)) / X_loc.shape[0]
                    worker_weights[rank] -= learning_rate * grad_loc

            step_counter += 1

            # Faza redukcji (Allreduce wag) co K krokow
            if sync_freq_k > 0 and step_counter % sync_freq_k == 0:
                avg_weights = np.mean(worker_weights, axis=0)
                for rank in range(n_workers):
                    worker_weights[rank][:] = avg_weights
                sync_count += 1

        # Synchronizacja koncowa po epoce
        if sync_freq_k > 0 and step_counter % sync_freq_k != 0:
            avg_weights = np.mean(worker_weights, axis=0)
            for rank in range(n_workers):
                worker_weights[rank][:] = avg_weights
            sync_count += 1

    return worker_weights[0], {"sync_count": sync_count}


def benchmark_mpi_scaling(
    X: np.ndarray,
    y: np.ndarray,
    process_counts: list[int] | None = None,
    epochs: int = 5,
    batch_size: int = 256,
    sync_freq_k: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Bada skalowalnosc rozproszonego Local SGD dla roznych liczb procesow p.
    Jesli w systemie dostepne jest mpiexec, uruchamia rzeczywiste procesy przez siec/IPC.
    W przeciwnym razie wykorzystuje symulator wieloprocesowy jako awaryjny fallback.
    """
    max_cores = os.cpu_count() or 4
    if process_counts is None:
        candidates = [1, 2, 4, 8]
        process_counts = [p for p in candidates if p <= max_cores]
        if 1 not in process_counts:
            process_counts.insert(0, 1)

    mpiexec_bin = shutil.which("mpiexec")
    backend_desc = "Rzeczywiste mpiexec" if mpiexec_bin else "Fallback symulator"
    print(f"\n--- Benchmark MPI / Local SGD Scaling ({backend_desc}, procesy: {process_counts}, K={sync_freq_k}) ---")
    results_by_procs: dict[int, dict[str, Any]] = {}

    for p in process_counts:
        samples: list[float] = []

        if mpiexec_bin:
            def run_mpi_real() -> float:
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
                    hist_file = tf.name
                try:
                    cmd = [
                        mpiexec_bin,
                        "--oversubscribe",
                        "-n", str(p),
                        sys.executable,
                        str(SRC_DIR / "main.py"),
                        "--mode", "mpi",
                        "--epochs", str(epochs),
                        "--batch-size", str(batch_size),
                        "--k", str(sync_freq_k),
                        "--seed", str(seed),
                        "--save-history", hist_file,
                        "--quiet",
                    ]
                    csgo_p = REPO_ROOT / "data" / "csgo_round_snapshots.csv"
                    if csgo_p.is_file():
                        cmd.extend(["--data", str(csgo_p)])

                    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
                    if res.returncode != 0:
                        raise RuntimeError(f"mpiexec zwrocil kod {res.returncode}:\n{res.stderr}")

                    with open(hist_file, "r", encoding="utf-8") as f:
                        hist_data = json.load(f)
                    total_t = float(sum(hist_data.get("epoch_times", [])))
                    return max(total_t, 0.001)
                finally:
                    if os.path.exists(hist_file):
                        os.remove(hist_file)

            try:
                # Rozgrzewka
                run_mpi_real()
                # 5 wlasciwych pomiarow
                for _ in range(5):
                    samples.append(run_mpi_real())
            except Exception as exc:
                print(f"[Warn] Blad uruchomienia mpiexec dla p={p} ({exc}), przelaczam na symulator.")
                samples = []

        if not samples:
            def run_sim() -> None:
                simulate_local_sgd_multiprocess(
                    X,
                    y,
                    n_workers=p,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=0.01,
                    sync_freq_k=sync_freq_k,
                    seed=seed,
                )
            _, samples = measure_median_time(run_sim, warmup_runs=1, measurement_runs=5)

        med_time = float(np.median(samples))
        print(f"Procesy p={p:2d} | Mediana: {med_time:.4f}s | Probki: {[round(s, 4) for s in samples]}")

        results_by_procs[p] = {
            "processes": p,
            "median_time": med_time,
            "samples": samples,
            "min_time": float(np.min(samples)),
            "max_time": float(np.max(samples)),
        }

    # Przyspieszenie i efektywnosc
    t1 = results_by_procs[process_counts[0]]["median_time"]
    for p in process_counts:
        tp = results_by_procs[p]["median_time"]
        speedup = t1 / tp if tp > 0 else 1.0
        efficiency = speedup / p
        results_by_procs[p]["speedup"] = float(speedup)
        results_by_procs[p]["efficiency"] = float(efficiency)

    return {
        "process_counts": process_counts,
        "results": {str(k): v for k, v in results_by_procs.items()},
        "baseline_time_t1": t1,
    }


def benchmark_batch_and_k_sensitivity(
    X_train: np.ndarray,
    y_train: np.ndarray,
    batch_sizes: list[int] | None = None,
    k_values: list[int] | None = None,
    target_accuracy: float = 0.85,
    max_epochs: int = 25,
    n_workers: int = 4,
    learning_rate: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Eksperyment: Wplyw wielkosci partii (batch size) i wspolczynnika synchronizacji K
    na 'Czas do 85% Accuracy' (lub innej zadanej dokladnosci).
    """
    if batch_sizes is None:
        batch_sizes = [64, 128, 256, 512]
    if k_values is None:
        k_values = [1, 2, 4, 8, 16]

    print(f"\n--- Eksperyment: Czas do {target_accuracy * 100:.0f}% Accuracy ---")
    print(f"Batch sizes: {batch_sizes} | K values: {k_values} | Workers: {n_workers}")

    n_samples, n_features = X_train.shape
    grid_results: list[dict[str, Any]] = []

    for bs in batch_sizes:
        for k in k_values:
            # Powtarzamy pomiary z rozgrzewka i mediana
            def train_and_track_time() -> tuple[float, int, float]:
                rng = np.random.default_rng(seed)
                worker_weights = [np.zeros(n_features, dtype=X_train.dtype) for _ in range(n_workers)]
                step_counter = 0

                t_start = time.perf_counter()
                reached_epoch = max_epochs
                final_acc = 0.0

                for epoch in range(1, max_epochs + 1):
                    indices = rng.permutation(n_samples)
                    X_shuffled = np.ascontiguousarray(X_train[indices])
                    y_shuffled = np.ascontiguousarray(y_train[indices])

                    for start_idx in range(0, n_samples, bs):
                        end_idx = start_idx + bs
                        X_batch = X_shuffled[start_idx:end_idx]
                        y_batch = y_shuffled[start_idx:end_idx]

                        for rank in range(n_workers):
                            X_loc, y_loc = split_batch_for_rank(X_batch, y_batch, rank, n_workers)
                            if X_loc.shape[0] > 0:
                                y_pred = sigmoid(X_loc @ worker_weights[rank])
                                grad_loc = (X_loc.T @ (y_pred - y_loc)) / X_loc.shape[0]
                                worker_weights[rank] -= learning_rate * grad_loc

                        step_counter += 1

                        if k > 0 and step_counter % k == 0:
                            avg_w = np.mean(worker_weights, axis=0)
                            for rank in range(n_workers):
                                worker_weights[rank][:] = avg_w

                    # Sprawdzenie dokladnosci po epoce na zsynchronizowanych wagach
                    avg_w = np.mean(worker_weights, axis=0)
                    probs = predict_proba(X_train, avg_w)
                    acc = compute_accuracy(y_train, probs)
                    final_acc = acc

                    if acc >= target_accuracy:
                        reached_epoch = epoch
                        break

                total_time = time.perf_counter() - t_start
                return total_time, reached_epoch, final_acc

            # Rozgrzewka
            train_and_track_time()

            # 5 pomiarow
            times: list[float] = []
            epochs_needed: list[int] = []
            accs: list[float] = []
            for _ in range(5):
                t_elapsed, ep, acc = train_and_track_time()
                times.append(t_elapsed)
                epochs_needed.append(ep)
                accs.append(acc)

            med_time = float(np.median(times))
            med_epoch = int(np.median(epochs_needed))
            best_acc = float(np.mean(accs))

            print(
                f"Batch={bs:3d}, K={k:2d} -> "
                f"Czas: {med_time:.4f}s | Epoki: {med_epoch:2d} | Acc: {best_acc * 100:.2f}%"
            )

            grid_results.append({
                "batch_size": bs,
                "sync_k": k,
                "median_time": med_time,
                "samples_time": times,
                "median_epochs": med_epoch,
                "accuracy": best_acc,
                "target_reached": best_acc >= target_accuracy,
            })

    return {
        "target_accuracy": target_accuracy,
        "batch_sizes": batch_sizes,
        "k_values": k_values,
        "grid_results": grid_results,
    }


def run_full_benchmark_suite(
    output_path: Path | None = None,
    use_synthetic_for_accuracy: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Glowna funkcja uruchamiajaca caly zestaw pomiarow i zapisujaca wyniki do pliku JSON.
    """
    if output_path is None:
        output_path = REPO_ROOT / "reports" / "benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("================================================================")
    print("ROZPOCZECIE BENCHMARKOW WYDAJNOSCIOWYCH (PRiR Issue #9)")
    print(f"Ziarno losowosci: {seed}")
    hw_info = get_hardware_info()
    print(f"Sprzet: {hw_info['processor']} | Rdzenie logiczne: {hw_info['cpu_count_logical']}")
    print("================================================================")

    # Dane do testow skalowania: sprawdzamy czy istnieje pelny zbior CS:GO
    csgo_path = REPO_ROOT / "data" / "csgo_round_snapshots.csv"
    if csgo_path.is_file():
        print(f"Ladowanie pelnego zbioru CS:GO z: {csgo_path}")
        X_raw, y_raw = load_and_preprocess_data(csgo_path, dtype=np.float32)
        X_tr, _, y_tr, _ = split_dataset(X_raw, y_raw, test_size=0.2, seed=seed)
        X_tr_scaled, _, _ = scale_features(X_tr, X_tr)
        X_scale = add_intercept(X_tr_scaled)
        y_scale = y_tr
        dataset_name = "CS:GO Round Snapshots (real dataset)"
    else:
        print("Brak pliku CSV CS:GO, uzywam zbioru syntetycznego.")
        X_scale, y_scale = generate_synthetic_dataset(n_samples=20000, n_features=30, seed=seed, dtype=np.float32)
        X_scale = add_intercept(X_scale)
        dataset_name = "Synthetic Dataset"

    print(f"Dane do skalowania: {X_scale.shape[0]} probek, {X_scale.shape[1]} cech")

    # Dane do eksperymentu 'Czas do 85% Accuracy' (zgodnie ze specyfikacja zadania)
    X_syn, y_syn = generate_synthetic_dataset(n_samples=15000, n_features=20, seed=seed, dtype=np.float32)
    X_syn_b = add_intercept(X_syn)

    # 1. Benchmark OpenMP / Numba
    openmp_data = benchmark_openmp_numba(
        X_scale,
        y_scale,
        thread_counts=[1, 2, 4, 8],
        epochs=3,
        batch_size=512,
        seed=seed,
    )

    # 2. Benchmark MPI Scaling
    mpi_data = benchmark_mpi_scaling(
        X_scale,
        y_scale,
        process_counts=[1, 2, 4, 8],
        epochs=3,
        batch_size=512,
        sync_freq_k=4,
        seed=seed,
    )

    # 3. Analiza Batch Size i parametru K na czas osiagniecia 85% accuracy
    batch_k_data = benchmark_batch_and_k_sensitivity(
        X_syn_b,
        y_syn,
        batch_sizes=[64, 128, 256, 512],
        k_values=[1, 2, 4, 8, 16],
        target_accuracy=0.85,
        max_epochs=20,
        n_workers=4,
        learning_rate=0.08,
        seed=seed,
    )

    full_report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": seed,
            "dataset_scaling": dataset_name,
            "hardware": hw_info,
            "note_cuda_synchronize": (
                "W srodowiskach CUDA kernele GPU sa asynchroniczne z perspektywy CPU. "
                "Do wiarygodnego pomiaru czasu kerneli konieczne jest wywolanie "
                "cuda.synchronize() bezposrednio przed i po pomiarze."
            ),
        },
        "openmp_benchmark": openmp_data,
        "mpi_benchmark": mpi_data,
        "batch_k_sensitivity": batch_k_data,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Pelne wyniki benchmarkow zapisano do: {output_path}")
    return full_report


if __name__ == "__main__":
    run_full_benchmark_suite()
