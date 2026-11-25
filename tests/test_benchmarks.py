import json
from pathlib import Path
import tempfile
import numpy as np
import pytest

from scripts.run_benchmarks import (
    benchmark_data_size_scaling,
    benchmark_openmp_numba,
    benchmark_sequential_baseline,
    measure_median_time,
    get_hardware_info,
)
from scripts.plot_results import (
    compute_amdahl_speedup,
    fit_parallel_fraction,
    plot_speedup_and_efficiency,
    plot_batch_and_k_analysis,
    plot_execution_time_comparison,
    plot_data_size_scaling,
    summarize_cuda_vs_seq,
)


def test_measure_median_time_discards_warmup_and_collects_5_samples():
    """
    Sprawdza czy funkcja pomiarowa wykonuje rozgrzewke i poprawnie zbiera
    dokladnie 5 probek oraz liczy z nich mediane.
    """
    call_counts = 0

    def mock_function():
        nonlocal call_counts
        call_counts += 1

    med_time, samples = measure_median_time(mock_function, warmup_runs=1, measurement_runs=5)

    # 1 rozgrzewka + 5 pomiarow = 6 wywolan funkcji
    assert call_counts == 6
    assert len(samples) == 5
    assert med_time >= 0.0
    assert np.isclose(med_time, np.median(samples))


def test_speedup_and_efficiency_mathematical_properties():
    """
    Weryfikuje wlasnosci matematyczne Przyspieszenia i Efektywnosci:
    - Dla 1 rdzenia S(1) = 1.0 oraz E(1) = 1.0 (lub 100%)
    - E(p) = S(p) / p
    """
    t1 = 10.0
    tp_values = {1: 10.0, 2: 5.5, 4: 3.0}

    for p, tp in tp_values.items():
        speedup = t1 / tp
        efficiency = speedup / p

        if p == 1:
            assert np.isclose(speedup, 1.0)
            assert np.isclose(efficiency, 1.0)
        else:
            assert speedup > 1.0
            assert efficiency <= 1.0
            assert np.isclose(efficiency, speedup / p)


def test_amdahl_law_curve_properties():
    """
    Weryfikuje czy Prawo Amdahla spelnia zalozenia teoretyczne:
    - S(1) = 1.0
    - Asymptota dla nieskonczonej liczby procesorow: lim_{p->inf} S(p) = 1 / (1 - f)
    - Monotoniczny wzrost S(p) ze wzrostem p.
    """
    f = 0.90  # 90% kodu zrownoleglone
    p_values = np.array([1, 2, 4, 8, 16, 1000])

    s_values = compute_amdahl_speedup(p_values, f)

    assert np.isclose(s_values[0], 1.0)
    assert np.all(np.diff(s_values) > 0), "S(p) musi monotonicznie rosnac"

    # Asymptota: 1 / (1 - 0.90) = 10.0
    p_large = np.array([100000])
    s_large = compute_amdahl_speedup(p_large, f)
    assert s_large[0] < 10.0
    assert np.isclose(s_large[0], 10.0, atol=0.001)


def test_hardware_info_returns_required_fields():
    """
    Sprawdza czy pobieranie informacji sprzetowych zwraca wymagane pola do legendy.
    """
    hw = get_hardware_info()
    assert "processor" in hw
    assert "cpu_count_logical" in hw
    assert hw["cpu_count_logical"] >= 1
    assert "system" in hw


def test_plot_generation_creates_valid_image_files(tmp_path: Path):
    """
    Testuje czy generator wykresow tworzy pliki PNG i czy maja one niezerowy rozmiar.
    """
    mock_data = {
        "metadata": {
            "seed": 42,
            "hardware": {"processor": "Test CPU", "cpu_count_logical": 4},
        },
        "openmp_benchmark": {
            "thread_counts": [1, 2, 4],
            "results": {
                "1": {"speedup": 1.0, "efficiency": 1.0, "median_time": 2.0},
                "2": {"speedup": 1.8, "efficiency": 0.9, "median_time": 1.1},
                "4": {"speedup": 3.2, "efficiency": 0.8, "median_time": 0.625},
            },
        },
        "mpi_benchmark": {
            "process_counts": [1, 2, 4],
            "results": {
                "1": {"speedup": 1.0, "efficiency": 1.0, "median_time": 2.2},
                "2": {"speedup": 1.7, "efficiency": 0.85, "median_time": 1.29},
                "4": {"speedup": 2.9, "efficiency": 0.725, "median_time": 0.76},
            },
        },
        "batch_k_sensitivity": {
            "target_accuracy": 0.85,
            "batch_sizes": [64, 128],
            "k_values": [1, 4],
            "grid_results": [
                {"batch_size": 64, "sync_k": 1, "median_time": 1.5},
                {"batch_size": 64, "sync_k": 4, "median_time": 0.9},
                {"batch_size": 128, "sync_k": 1, "median_time": 1.2},
                {"batch_size": 128, "sync_k": 4, "median_time": 0.7},
            ],
        },
    }

    out_speedup = tmp_path / "speedup_efficiency.png"
    out_batch_k = tmp_path / "batch_k_analysis.png"

    plot_speedup_and_efficiency(mock_data, output_path=out_speedup)
    plot_batch_and_k_analysis(mock_data, output_path=out_batch_k)

    assert out_speedup.is_file(), "Wykres przyspieszenia powinien zostac utworzony"
    assert out_speedup.stat().st_size > 1000, "Plik PNG nie moze byc pusty"

    assert out_batch_k.is_file(), "Wykres Batch-K powinien zostac utworzony"
    assert out_batch_k.stat().st_size > 1000, "Plik PNG nie moze byc pusty"


def test_data_size_scaling_runner_and_plot(tmp_path: Path):
    """
    Sprawdza dzialanie benchmark_data_size_scaling na malych danych
    oraz generowanie wykresu plot_data_size_scaling (z danymi CUDA i bez nich).
    """
    res = benchmark_data_size_scaling(
        sample_sizes=[200, 400],
        n_features=8,
        epochs=2,
        batch_size=64,
        n_threads=2,
        n_mpi_procs=2,
    )
    assert len(res["results"]) == 2
    assert res["results"][0]["n_samples"] == 200
    assert res["results"][0]["seq_epoch_time"] > 0
    assert res["results"][0]["speedup_numba"] > 0
    assert res["results"][0]["speedup_mpi"] > 0

    # Test generowania wykresu (wersja z danymi CUDA w mocku)
    res_with_cuda = dict(res)
    res_with_cuda["cuda_device_name"] = "Mock T4"
    res_with_cuda["results"] = [
        dict(r, cuda_epoch_time=r["seq_epoch_time"] * 0.2, speedup_cuda=5.0)
        for r in res["results"]
    ]
    out_plot = tmp_path / "data_size_scaling.png"
    plot_data_size_scaling({"data_size_scaling": res_with_cuda}, output_path=out_plot)
    assert out_plot.is_file()
    assert out_plot.stat().st_size > 1000


def test_sequential_baseline_and_speedup_vs_seq(tmp_path: Path):
    """
    Weryfikuje pomiar T_seq (benchmark_sequential_baseline), wyznaczanie speedup_vs_seq
    i efficiency_vs_seq z zachowaniem skalowania wzgledem T_1 oraz generowanie wykresow.
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((500, 8), dtype=np.float32)
    y = rng.integers(0, 2, size=500).astype(np.float32)

    seq = benchmark_sequential_baseline(X, y, epochs=2, batch_size=64, seed=42, warmup_runs=1, measurement_runs=3)
    assert seq["median_time"] > 0
    assert len(seq["samples"]) == 3
    assert "timing_scope" in seq

    omp = benchmark_openmp_numba(
        X, y, thread_counts=[1, 2], epochs=2, batch_size=64, seed=42, baseline_seq_time=seq["median_time"]
    )
    assert omp["timing_scope"] == seq["timing_scope"]
    assert np.isclose(omp["baseline_time_seq"], seq["median_time"])
    for t_str, row in omp["results"].items():
        t = int(t_str)
        assert np.isclose(row["speedup_vs_seq"], seq["median_time"] / row["median_time"])
        assert np.isclose(row["efficiency_vs_seq"], row["speedup_vs_seq"] / t)
        assert "speedup" in row and "efficiency" in row

    mock_mpi = {
        "process_counts": [1, 2],
        "baseline_time_t1": 0.02,
        "baseline_time_seq": seq["median_time"],
        "results": {
            "1": {"median_time": 0.02, "speedup": 1.0, "efficiency": 1.0, "speedup_vs_seq": seq["median_time"] / 0.02, "efficiency_vs_seq": seq["median_time"] / 0.02},
            "2": {"median_time": 0.015, "speedup": 0.02 / 0.015, "efficiency": (0.02 / 0.015) / 2, "speedup_vs_seq": seq["median_time"] / 0.015, "efficiency_vs_seq": (seq["median_time"] / 0.015) / 2},
        },
    }
    report = {
        "metadata": {"seed": 42, "hardware": {"processor": "Test", "cpu_count_logical": 4}},
        "sequential_baseline": seq,
        "openmp_benchmark": omp,
        "mpi_benchmark": mock_mpi,
    }
    out_se = tmp_path / "se.png"
    out_time = tmp_path / "time.png"
    plot_speedup_and_efficiency(report, output_path=out_se)
    plot_execution_time_comparison(report, output_path=out_time)
    assert out_se.is_file() and out_se.stat().st_size > 1000
    assert out_time.is_file() and out_time.stat().st_size > 1000


def test_cuda_summary_preserves_environment_metadata():
    """
    Sprawdza, czy summarize_cuda_vs_seq przenosi metadane sprzetowe (GPU, CPU,
    wersje Numba/CUDA, data) z plikow historii do podsumowania cuda_results.json.
    """
    meta = {
        "timestamp": "2026-11-04 14:27:00",
        "environment": "Google Colab (NVIDIA T4 GPU runtime)",
        "gpu_model": "NVIDIA Tesla T4",
        "cuda_version": "12.4",
        "cpu": "Intel Xeon",
        "numba_version": "0.60.0",
    }
    seq_hist = {"epoch_times": [0.04, 0.042, 0.041], "accuracy": [0.73, 0.74, 0.75], "loss": [0.55, 0.50, 0.48], "metadata": meta}
    cuda_hist = {"epoch_times": [1.40, 0.10, 0.105], "accuracy": [0.73, 0.74, 0.75], "loss": [0.55, 0.50, 0.48], "metadata": meta}

    summary = summarize_cuda_vs_seq(seq_hist, cuda_hist)
    assert "metadata" in summary
    assert summary["metadata"]["gpu_model"] == "NVIDIA Tesla T4"
    assert summary["metadata"]["cuda_version"] == "12.4"
    assert summary["metadata"]["cpu"] == "Intel Xeon"
    assert summary["metadata"]["numba_version"] == "0.60.0"
    assert summary["metadata"]["timestamp"] == "2026-11-04 14:27:00"
