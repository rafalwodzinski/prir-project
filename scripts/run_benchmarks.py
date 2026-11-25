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

REPO_ROOT = Path(__file__).resolve().parent.parent

from numba import get_num_threads, set_num_threads
from src.data_loader import add_intercept, prepare_dataset
from src.main import collect_environment_metadata
from src.sequential_sgd import compute_accuracy, generate_synthetic_dataset, predict_proba, train_sequential_sgd
from src.numba_sgd import compute_gradient_numba, train_numba_sgd
from src.model_mpi import (
    MPI,
    run_local_sgd_epoch_simulated,
    split_batch_for_rank,
    train_local_sgd_mpi,
)


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
    Pobiera pelne informacje o srodowisku sprzetowym (CPU, GPU z nvidia-smi,
    wersje Python/NumPy/Numba/CUDA) do metadanych JSON i legendy wykresow.
    """
    env = collect_environment_metadata()
    return {
        "processor": platform.processor() or "CPU x86_64",
        "cpu_count_logical": env["cpu_count_logical"],
        "python_version": env["python_version"],
        "numpy_version": env["numpy_version"],
        "numba_version": env["numba_version"],
        "gpu_model": env["gpu_model"],
        "nvidia_driver": env["nvidia_driver"],
        "cuda_version": env["cuda_version"],
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


TIMING_SCOPE_DESC = "sum(epoch_times) po rozgrzewce JIT (wylacznie matematyka SGD + komunikacja, bez I/O i ewaluacji)"


def benchmark_sequential_baseline(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 5,
    batch_size: int = 256,
    seed: int = 42,
    warmup_runs: int = 1,
    measurement_runs: int = 5,
) -> dict[str, Any]:
    """
    Mierzy czas wykonania bazowej wersji sekwencyjnej (NumPy, 1 watek)
    jako sume czasow epok SGD (bez ewaluacji), w pelni spojna z OpenMP i MPI.
    Zwraca slownik z mediana T_seq oraz probkami.
    """
    print(f"\n--- Benchmark Sequential Baseline (NumPy 1T, epochs={epochs}, batch={batch_size}) ---")

    def run_once() -> float:
        _, hist = train_sequential_sgd(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=0.01,
            seed=seed,
            verbose=False,
        )
        return float(sum(hist.get("epoch_times", [])))

    for _ in range(warmup_runs):
        run_once()

    samples = [run_once() for _ in range(measurement_runs)]
    med_time = float(np.median(samples))
    print(f"Sequential T_seq | Mediana: {med_time:.4f}s | Probki: {[round(s, 4) for s in samples]}")

    return {
        "median_time": med_time,
        "samples": samples,
        "min_time": float(np.min(samples)),
        "max_time": float(np.max(samples)),
        "epochs": epochs,
        "batch_size": batch_size,
        "timing_scope": TIMING_SCOPE_DESC,
    }


def benchmark_openmp_numba(
    X: np.ndarray,
    y: np.ndarray,
    thread_counts: list[int] | None = None,
    epochs: int = 5,
    batch_size: int = 256,
    seed: int = 42,
    baseline_seq_time: float | None = None,
) -> dict[str, Any]:
    """
    Bada skalowalnosc wielowatkowej implementacji Numba (odpowiednik OpenMP na CPU).
    Mierzy czasy epok dla roznych liczb watkow z uzyciem set_num_threads.
    Liczy skalowanie wzgledem T_1 oraz (gdy podano baseline_seq_time) wzgledem T_seq.
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

        def run_once() -> float:
            _, hist = train_numba_sgd(
                X,
                y,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=0.01,
                seed=seed,
                verbose=False,
            )
            return float(sum(hist.get("epoch_times", [])))

        # 1 rozgrzewka + 5 pomiarow sumy czasow epok (spojne z T_seq i MPI)
        run_once()
        samples = [run_once() for _ in range(5)]
        med_time = float(np.median(samples))
        print(f"Watki p={t:2d} | Mediana: {med_time:.4f}s | Probki: {[round(s, 4) for s in samples]}")

        results_by_threads[t] = {
            "threads": t,
            "median_time": med_time,
            "samples": samples,
            "min_time": float(np.min(samples)),
            "max_time": float(np.max(samples)),
        }

    # Obliczamy przyspieszenie S(p) = T_1 / T_p oraz efektywnosc E(p) = S(p) / p
    # a takze przyspieszenie i efektywnosc wzgledem bazowego T_seq (NumPy)
    t1 = results_by_threads[thread_counts[0]]["median_time"]
    for t in thread_counts:
        tp = results_by_threads[t]["median_time"]
        speedup = t1 / tp if tp > 0 else 1.0
        efficiency = speedup / t
        results_by_threads[t]["speedup"] = float(speedup)
        results_by_threads[t]["efficiency"] = float(efficiency)
        if baseline_seq_time is not None:
            speedup_seq = baseline_seq_time / tp if tp > 0 else 0.0
            results_by_threads[t]["speedup_vs_seq"] = float(speedup_seq)
            results_by_threads[t]["efficiency_vs_seq"] = float(speedup_seq / t)

    result: dict[str, Any] = {
        "thread_counts": thread_counts,
        "results": {str(k): v for k, v in results_by_threads.items()},
        "baseline_time_t1": t1,
        "timing_scope": TIMING_SCOPE_DESC,
    }
    if baseline_seq_time is not None:
        result["baseline_time_seq"] = float(baseline_seq_time)
    return result


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
    epoch_times: list[float] = []

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        indices = rng.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        step_counter, epoch_syncs = run_local_sgd_epoch_simulated(
            X_shuffled,
            y_shuffled,
            worker_weights,
            batch_size=batch_size,
            learning_rate=learning_rate,
            sync_freq_k=sync_freq_k,
            step_counter=step_counter,
        )
        sync_count += epoch_syncs
        epoch_times.append(time.perf_counter() - t0)

    return worker_weights[0], {"sync_count": sync_count, "epoch_times": epoch_times}


def benchmark_mpi_scaling(
    X: np.ndarray,
    y: np.ndarray,
    process_counts: list[int] | None = None,
    epochs: int = 5,
    batch_size: int = 256,
    sync_freq_k: int = 4,
    seed: int = 42,
    baseline_seq_time: float | None = None,
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
    backend_used = "mpiexec" if mpiexec_bin else "simulator"
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
                        "-m", "src.main",
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
                backend_used = "simulator"
                samples = []

        if not samples:
            def run_sim() -> float:
                _, hist = simulate_local_sgd_multiprocess(
                    X,
                    y,
                    n_workers=p,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=0.01,
                    sync_freq_k=sync_freq_k,
                    seed=seed,
                )
                return float(sum(hist.get("epoch_times", [])))
            # 1 rozgrzewka + 5 pomiarow sumy epoch_times (spojne z mpiexec, T_seq i Numba)
            run_sim()
            samples = [run_sim() for _ in range(5)]

        med_time = float(np.median(samples))
        print(f"Procesy p={p:2d} | Mediana: {med_time:.4f}s | Probki: {[round(s, 4) for s in samples]}")

        results_by_procs[p] = {
            "processes": p,
            "median_time": med_time,
            "samples": samples,
            "min_time": float(np.min(samples)),
            "max_time": float(np.max(samples)),
        }

    # Przyspieszenie i efektywnosc wzgledem T_1 oraz (opcjonalnie) wzgledem T_seq
    t1 = results_by_procs[process_counts[0]]["median_time"]
    for p in process_counts:
        tp = results_by_procs[p]["median_time"]
        speedup = t1 / tp if tp > 0 else 1.0
        efficiency = speedup / p
        results_by_procs[p]["speedup"] = float(speedup)
        results_by_procs[p]["efficiency"] = float(efficiency)
        if baseline_seq_time is not None:
            speedup_seq = baseline_seq_time / tp if tp > 0 else 0.0
            results_by_procs[p]["speedup_vs_seq"] = float(speedup_seq)
            results_by_procs[p]["efficiency_vs_seq"] = float(speedup_seq / p)

    result: dict[str, Any] = {
        "process_counts": process_counts,
        "backend": backend_used,
        "timing_scope": TIMING_SCOPE_DESC,
        "results": {str(k): v for k, v in results_by_procs.items()},
        "baseline_time_t1": t1,
    }
    if baseline_seq_time is not None:
        result["baseline_time_seq"] = float(baseline_seq_time)
    return result


def benchmark_batch_and_k_sensitivity(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
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
    Prog 74% wynika z liniowego sufitu regresji logistycznej na CS:GO (~75% Accuracy).
    Gdy podano (X_test, y_test), po osiagnieciu progu wyznaczana jest rowniez dokladnosc testowa (test_accuracy).
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
            def train_and_track_time() -> tuple[float, int, float, float | None]:
                rng = np.random.default_rng(seed)
                worker_weights = [np.zeros(n_features, dtype=X_train.dtype) for _ in range(n_workers)]
                step_counter = 0

                t_start = time.perf_counter()
                reached_epoch = max_epochs
                final_acc = 0.0

                for epoch in range(1, max_epochs + 1):
                    indices = rng.permutation(n_samples)
                    X_shuffled = X_train[indices]
                    y_shuffled = y_train[indices]

                    step_counter, _ = run_local_sgd_epoch_simulated(
                        X_shuffled,
                        y_shuffled,
                        worker_weights,
                        batch_size=bs,
                        learning_rate=learning_rate,
                        sync_freq_k=k,
                        step_counter=step_counter,
                    )

                    # Sprawdzenie dokladnosci po epoce na zsynchronizowanych wagach (worker_weights sa juz usrednione)
                    probs = predict_proba(X_train, worker_weights[0])
                    acc = compute_accuracy(y_train, probs)
                    final_acc = acc

                    if acc >= target_accuracy:
                        reached_epoch = epoch
                        break

                total_time = time.perf_counter() - t_start

                # Ewaluacja na zbiorze testowym poza pomiarem czasu treningu
                test_acc: float | None = None
                if X_test is not None and y_test is not None:
                    test_probs = predict_proba(X_test, worker_weights[0])
                    test_acc = compute_accuracy(y_test, test_probs)

                return total_time, reached_epoch, final_acc, test_acc

            # Rozgrzewka
            train_and_track_time()

            # 5 pomiarow
            times: list[float] = []
            epochs_needed: list[int] = []
            accs: list[float] = []
            test_accs: list[float] = []
            for _ in range(5):
                t_elapsed, ep, acc, t_acc = train_and_track_time()
                times.append(t_elapsed)
                epochs_needed.append(ep)
                accs.append(acc)
                if t_acc is not None:
                    test_accs.append(t_acc)

            med_time = float(np.median(times))
            med_epoch = int(np.median(epochs_needed))
            best_acc = float(np.mean(accs))
            mean_test_acc = float(np.mean(test_accs)) if test_accs else None

            test_str = f" | Test Acc: {mean_test_acc * 100:.2f}%" if mean_test_acc is not None else ""
            print(
                f"Batch={bs:3d}, K={k:2d} -> "
                f"Czas: {med_time:.4f}s | Epoki: {med_epoch:2d} | Train Acc: {best_acc * 100:.2f}%{test_str}"
            )

            entry: dict[str, Any] = {
                "batch_size": bs,
                "sync_k": k,
                "median_time": med_time,
                "samples_time": times,
                "median_epochs": med_epoch,
                "accuracy": best_acc,
                "train_accuracy": best_acc,
                "target_reached": best_acc >= target_accuracy,
            }
            if mean_test_acc is not None:
                entry["test_accuracy"] = mean_test_acc
            grid_results.append(entry)

    return {
        "execution_mode": "simulated_workers",
        "n_workers": n_workers,
        "note": (
            "Symulacja wieloprocesowego Local SGD w jednym procesie "
            "(wspolna logika run_local_sgd_epoch_simulated z src/model_mpi.py) – "
            "mierzy matematyczny wplyw Batch Size i K bez narzutu startu mpiexec. "
            "Prog 74% Accuracy wynika z liniowego sufitu regresji logistycznej na CS:GO (~75%)."
        ),
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
        from src.cuda_sgd import train_cuda_sgd
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
                    "-m", "src.main",
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
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": get_hardware_info(),
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
    skip_data_scaling: bool = False,
) -> dict[str, Any]:
    """
    Glowna funkcja uruchamiajaca caly zestaw pomiarow i zapisujaca wyniki do pliku JSON.
    """
    if output_path is None:
        output_path = REPO_ROOT / "reports" / "benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_data_size_scaling: dict[str, Any] | None = None
    if skip_data_scaling and output_path.is_file():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data_size_scaling = json.load(f).get("data_size_scaling")
        except Exception:
            existing_data_size_scaling = None

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

    X_scale, X_scale_test, y_scale, y_scale_test = prepare_dataset(
        csgo_path,
        test_size=0.2,
        seed=seed,
        dtype=np.float32,
        fallback=lambda: generate_synthetic_dataset(n_samples=20000, n_features=30, seed=seed, dtype=np.float32),
    )

    print(f"Dane do eksperymentow: train={X_scale.shape[0]}, test={X_scale_test.shape[0]}, cech={X_scale.shape[1]}")

    # 0. Benchmark Sequential Baseline (NumPy 1T) - punkt odniesienia T_seq
    seq_data = benchmark_sequential_baseline(
        X_scale,
        y_scale,
        epochs=3,
        batch_size=512,
        seed=seed,
    )
    t_seq = seq_data["median_time"]

    # 1. Benchmark OpenMP / Numba
    openmp_data = benchmark_openmp_numba(
        X_scale,
        y_scale,
        thread_counts=[1, 2, 4, 8],
        epochs=3,
        batch_size=512,
        seed=seed,
        baseline_seq_time=t_seq,
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
        baseline_seq_time=t_seq,
    )

    # 3. Analiza Batch Size i parametru K na czas osiagniecia 74% accuracy (dane CS:GO, ewaluacja train + test)
    batch_k_data = benchmark_batch_and_k_sensitivity(
        X_scale,
        y_scale,
        X_test=X_scale_test,
        y_test=y_scale_test,
        batch_sizes=[64, 128, 256, 512],
        k_values=[1, 2, 4, 8, 16],
        target_accuracy=0.74,
        max_epochs=20,
        n_workers=4,
        learning_rate=0.003,
        seed=seed,
    )

    # 4. Skalowanie wzgledem rozmiaru danych (seq, Numba, MPI, CUDA)
    if skip_data_scaling:
        data_size_data = existing_data_size_scaling
        print("\n[Info] Pominieto data_size_scaling (--skip-data-scaling); zachowano istniejace wyniki.")
    else:
        data_size_data = benchmark_data_size_scaling(
            sample_sizes=sample_sizes,
            n_features=data_scaling_features,
            epochs=4,
            batch_size=512,
            seed=seed,
        )

    full_report: dict[str, Any] = {
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
            "timing_methodology": {
                "scope": TIMING_SCOPE_DESC,
                "sequential": "Suma epoch_times (tasowanie + mini-batch SGD NumPy/BLAS, bez I/O i bez ewaluacji predict_proba), mediana z 5 prob po 1 rozgrzewce.",
                "openmp_numba": "Suma epoch_times (tasowanie + kernel @njit(parallel=True) prange + redukcja, bez kompilacji JIT, I/O i ewaluacji), mediana z 5 prob po 1 rozgrzewce.",
                "mpi": "Suma epoch_times procesu rank=0 (tasowanie + lokalne SGD NumPy + Allreduce co K krokow, bez startu mpiexec, I/O i ewaluacji), mediana z 5 prob po 1 rozgrzewce.",
                "cuda": "Czas pojedynczej epoki otoczony cuda.synchronize() (przeslanie indeksow int32 + 3 kernele GPU, bez transferu macierzy X do VRAM, bez kompilacji JIT z 1. epoki i bez ewaluacji).",
            },
        },
        "sequential_baseline": seq_data,
        "openmp_benchmark": openmp_data,
        "mpi_benchmark": mpi_data,
        "batch_k_sensitivity": batch_k_data,
    }
    if data_size_data is not None:
        full_report["data_size_scaling"] = data_size_data

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
        "--skip-data-scaling",
        action="store_true",
        help="Pomija eksperyment data_size_scaling i zachowuje istniejace wyniki z JSON (np. z Colaba)",
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
            skip_data_scaling=args.skip_data_scaling,
        )


if __name__ == "__main__":
    main()
