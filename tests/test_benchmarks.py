import json
from pathlib import Path
import tempfile
import numpy as np
import pytest

from scripts.run_benchmarks import measure_median_time, get_hardware_info
from scripts.plot_results import compute_amdahl_speedup, fit_parallel_fraction, plot_speedup_and_efficiency, plot_batch_and_k_analysis


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
