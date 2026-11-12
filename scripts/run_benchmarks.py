import argparse
from dataclasses import asdict, dataclass
import gc
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
from data_loader import add_intercept, prepare_dataset
from sequential_sgd import compute_accuracy, generate_synthetic_dataset, predict_proba, train_sequential_sgd
from numba_sgd import compute_gradient_numba, train_numba_sgd
from model_mpi import split_batch_for_rank, train_local_sgd_mpi, MPI
from math_utils import sigmoid


def _mpi_subprocess_env() -> dict[str, str]:
    """
    Zwraca srodowisko dla procesow potomnych mpiexec.
    W kontenerach (np. Google Colab) skrypt dziala jako root, wiec OpenMPI wymaga flagi zezwolenia.
    """
    env = os.environ.copy()
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
        env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    return env


def has_physical_cuda_gpu() -> tuple[bool, str | None]:
    """
    Sprawdza, czy dostepna jest fizyczna karta NVIDIA CUDA (z pominięciem CUDASIM).
    Musi byc wywolane przed zaimportowaniem cuda_sgd, aby nie wymuszac CUDASIM.
    """
    if os.environ.get("NUMBA_ENABLE_CUDASIM") == "1":
        return False, None
    try:
        from numba import cuda
        if cuda.is_available():
            device = cuda.get_current_device()
            name = device.name.decode("utf-8", errors="replace") if isinstance(device.name, bytes) else str(device.name)
            return True, name
    except Exception:
        pass
    return False, None


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

                    res = subprocess.run(cmd, cwd=REPO_ROOT, env=_mpi_subprocess_env(), capture_output=True, text=True, timeout=60)
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
    target_accuracy: float = 0.74,
    max_epochs: int = 25,
    n_workers: int = 4,
    learning_rate: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Eksperyment: Wplyw wielkosci partii (batch size) i wspolczynnika synchronizacji K
    na 'Czas do 74% Accuracy' (lub innej zadanej dokladnosci).
    Dane: zbior treningowy CS:GO (dokladnosc liczona na zbiorze treningowym, tak jak w history['accuracy']).
    Jesli prog nie zostanie osiagniety, trening konczy sie po max_epochs (target_reached=False).
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


def _median_epoch_time(epoch_times: list[float]) -> float:
    """
    Zwraca mediane czasu epok z pominieciem pierwszej epoki (rozgrzewka JIT/cache),
    jesli dostepne sa co najmniej 2 epoki.
    """
    if not epoch_times:
        return 0.0
    relevant = epoch_times[1:] if len(epoch_times) > 1 else epoch_times
    return float(np.median(relevant))


def benchmark_data_size_scaling(
    sample_sizes: list[int] | None = None,
    n_features: int = 100,
    epochs: int = 4,
    batch_size: int = 512,
    n_threads: int | None = None,
    n_mpi_procs: int = 4,
    sync_freq_k: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Eksperyment skalowania wzgledem rozmiaru danych (n_samples od 100k do 5M).
    Porownuje mediane czasu epoki (bez 1. epoki rozgrzewkowej) dla:
    - Sequential (CPU, 1 watek)
    - Numba / OpenMP (CPU wielowatkowy)
    - MPI (Local SGD na wielu procesach przez mpiexec, lub fallback)
    - CUDA (GPU - uruchamiane tylko gdy dostepne jest fizyczne GPU NVIDIA)
    """
    if sample_sizes is None:
        sample_sizes = [100_000, 500_000, 1_000_000, 5_000_000]

    max_cores = os.cpu_count() or 4
    if n_threads is None:
        n_threads = min(8, max_cores)

    mpiexec_bin = shutil.which("mpiexec")
    cuda_ok, cuda_device_name = has_physical_cuda_gpu()
    train_cuda_fn = None
    if cuda_ok:
        from cuda_sgd import train_cuda_sgd
        train_cuda_fn = train_cuda_sgd

    print(
        f"\n--- Eksperyment: Skalowanie wzgledem rozmiaru danych "
        f"(sizes={sample_sizes}, features={n_features}, epochs={epochs}, batch={batch_size}) ---"
    )
    print(
        f"Konfiguracja: Numba={n_threads}T | "
        f"MPI={n_mpi_procs}P ({'mpiexec' if mpiexec_bin else 'symulator'}) | "
        f"CUDA={cuda_device_name if cuda_ok else 'brak fizycznego GPU (pominieto)'}"
    )

    rows: list[dict[str, Any]] = []
    mpi_backend_used = "mpiexec" if mpiexec_bin else "simulator"

    for n_samples in sample_sizes:
        print(f"\n[Rozmiar n={n_samples:,}] Generowanie danych...")
        X_syn, y_syn = generate_synthetic_dataset(
            n_samples=n_samples,
            n_features=n_features,
            seed=seed,
            dtype=np.float32,
        )
        X_data = add_intercept(X_syn)
        del X_syn
        mem_mb = float((X_data.nbytes + y_syn.nbytes) / (1024 ** 2))

        # 1. Sequential
        _, hist_seq = train_sequential_sgd(
            X_data,
            y_syn,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=0.01,
            seed=seed,
            verbose=False,
        )
        t_seq = _median_epoch_time(hist_seq["epoch_times"])

        # 2. Numba (OpenMP)
        set_num_threads(n_threads)
        _, hist_numba = train_numba_sgd(
            X_data,
            y_syn,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=0.01,
            seed=seed,
            verbose=False,
        )
        t_numba = _median_epoch_time(hist_numba["epoch_times"])

        # 3. MPI
        t_mpi: float | None = None
        if mpiexec_bin:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
                hist_file = tf.name
            try:
                cmd = [
                    mpiexec_bin,
                    "--oversubscribe",
                    "-n", str(n_mpi_procs),
                    sys.executable,
                    str(SRC_DIR / "main.py"),
                    "--mode", "mpi",
                    "--synthetic-samples", str(n_samples),
                    "--synthetic-features", str(n_features),
                    "--test-size", "0",
                    "--epochs", str(epochs),
                    "--batch-size", str(batch_size),
                    "--k", str(sync_freq_k),
                    "--seed", str(seed),
                    "--save-history", hist_file,
                    "--quiet",
                ]
                res = subprocess.run(cmd, cwd=REPO_ROOT, env=_mpi_subprocess_env(), capture_output=True, text=True, timeout=300)
                if res.returncode != 0:
                    raise RuntimeError(f"mpiexec error {res.returncode}: {res.stderr}")
                with open(hist_file, "r", encoding="utf-8") as f:
                    hist_mpi = json.load(f)
                t_mpi = _median_epoch_time(hist_mpi.get("epoch_times", []))
            except Exception as exc:
                print(f"[Warn] Blad mpiexec dla n={n_samples}: {exc}. Przelaczam na symulator.")
                mpi_backend_used = "simulator"
            finally:
                if os.path.exists(hist_file):
                    os.remove(hist_file)

        if t_mpi is None:
            t0 = time.perf_counter()
            simulate_local_sgd_multiprocess(
                X_data,
                y_syn,
                n_workers=n_mpi_procs,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=0.01,
                sync_freq_k=sync_freq_k,
                seed=seed,
            )
            t_mpi = (time.perf_counter() - t0) / epochs

        # 4. CUDA (tylko jesli jest fizyczne GPU)
        t_cuda: float | None = None
        if cuda_ok and train_cuda_fn is not None:
            _, hist_cuda = train_cuda_fn(
                X_data,
                y_syn,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=0.01,
                seed=seed,
                verbose=False,
            )
            t_cuda = _median_epoch_time(hist_cuda["epoch_times"])

        speedup_numba = t_seq / t_numba if t_numba > 0 else 0.0
        speedup_mpi = t_seq / t_mpi if t_mpi > 0 else 0.0
        speedup_cuda = (t_seq / t_cuda) if (t_cuda is not None and t_cuda > 0) else None

        cuda_str = f"{t_cuda:.4f}s ({speedup_cuda:.2f}x)" if t_cuda is not None else "N/A"
        print(
            f"n={n_samples:8d} ({mem_mb:6.1f} MB) | "
            f"Seq: {t_seq:.4f}s | Numba({n_threads}T): {t_numba:.4f}s ({speedup_numba:.2f}x) | "
            f"MPI({n_mpi_procs}P): {t_mpi:.4f}s ({speedup_mpi:.2f}x) | CUDA: {cuda_str}"
        )

        rows.append({
            "n_samples": int(n_samples),
            "n_features": int(n_features + 1),
            "memory_mb": round(mem_mb, 2),
            "seq_epoch_time": t_seq,
            "numba_epoch_time": t_numba,
            "mpi_epoch_time": t_mpi,
            "cuda_epoch_time": t_cuda,
            "speedup_numba": speedup_numba,
            "speedup_mpi": speedup_mpi,
            "speedup_cuda": speedup_cuda,
        })

        del X_data, y_syn
        gc.collect()

    return {
        "sample_sizes": sample_sizes,
        "n_features": n_features + 1,
        "epochs": epochs,
        "batch_size": batch_size,
        "numba_threads": n_threads,
        "mpi_processes": n_mpi_procs,
        "mpi_backend": mpi_backend_used,
        "cuda_available": cuda_ok,
        "cuda_device_name": cuda_device_name,
        "results": rows,
    }


def run_full_benchmark_suite(
    output_path: Path | None = None,
    use_synthetic_for_accuracy: bool = True,
    seed: int = 42,
    sample_sizes: list[int] | None = None,
    data_scaling_features: int = 100,
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

    # Jedna wspolna macierz treningowa CS:GO (lub syntetyczna, gdy brak pliku) dla wszystkich eksperymentow
    csgo_path = REPO_ROOT / "data" / "csgo_round_snapshots.csv"
    if csgo_path.is_file():
        print(f"Ladowanie pelnego zbioru CS:GO z: {csgo_path}")
        dataset_name = "CS:GO Round Snapshots (real dataset)"
    else:
        print("Brak pliku CSV CS:GO, uzywam zbioru syntetycznego.")
        dataset_name = "Synthetic Dataset"

    X_scale, _, y_scale, _ = prepare_dataset(
        csgo_path,
        test_size=0.2,
        seed=seed,
        dtype=np.float32,
        fallback=lambda: generate_synthetic_dataset(n_samples=20000, n_features=30, seed=seed, dtype=np.float32),
    )

    print(f"Dane do eksperymentow: {X_scale.shape[0]} probek, {X_scale.shape[1]} cech")

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

    # 3. Analiza Batch Size i parametru K na czas osiagniecia 74% accuracy (dane CS:GO)
    batch_k_data = benchmark_batch_and_k_sensitivity(
        X_scale,
        y_scale,
        batch_sizes=[64, 128, 256, 512],
        k_values=[1, 2, 4, 8, 16],
        target_accuracy=0.74,
        max_epochs=20,
        n_workers=4,
        learning_rate=0.003,
        seed=seed,
    )

    # 4. Skalowanie wzgledem rozmiaru danych (seq, Numba, MPI, CUDA)
    data_size_data = benchmark_data_size_scaling(
        sample_sizes=sample_sizes,
        n_features=data_scaling_features,
        epochs=4,
        batch_size=512,
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
        "data_size_scaling": data_size_data,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Pelne wyniki benchmarkow zapisano do: {output_path}")
    return full_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmarki wydajnosciowe PRiR")
    parser.add_argument(
        "--only-data-scaling",
        action="store_true",
        help="Uruchamia tylko eksperyment skalowania rozmiaru danych i aktualizuje benchmark_results.json",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100_000, 500_000, 1_000_000, 5_000_000],
        help="Lista rozmiarow n_samples do eksperymentu skalowania danych",
    )
    parser.add_argument(
        "--features",
        type=int,
        default=100,
        help="Liczba cech syntetycznych w eksperymencie skalowania danych",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Liczba epok w eksperymencie skalowania danych (1. epoka odrzucana jako warm-up)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Wielkosc mini-batcha w eksperymencie skalowania danych",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "reports" / "benchmark_results.json"),
        help="Sciezka do pliku wynikowego JSON",
    )
    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.only_data_scaling:
        data_size_data = benchmark_data_size_scaling(
            sample_sizes=args.sizes,
            n_features=args.features,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        if output_path.is_file():
            with open(output_path, "r", encoding="utf-8") as f:
                report = json.load(f)
        else:
            report = {"metadata": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "hardware": get_hardware_info()}}
        report["data_size_scaling"] = data_size_data
        report.setdefault("metadata", {})["data_scaling_updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Zaktualizowano data_size_scaling w: {output_path}")
    else:
        run_full_benchmark_suite(
            output_path=output_path,
            sample_sizes=args.sizes,
            data_scaling_features=args.features,
        )


if __name__ == "__main__":
    main()
