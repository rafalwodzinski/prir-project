import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
import numpy as np

# Zabezpieczenie katalogu konfiguracyjnego matplotliba
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "figures"
REPORTS_DIR = REPO_ROOT / "reports"


def compute_amdahl_speedup(p_values: np.ndarray, parallel_fraction_f: float) -> np.ndarray:
    """
    Teoretyczne przyspieszenie wedlug Prawa Amdahla:
    S(p) = 1 / ((1 - f) + f / p)
    """
    f = np.clip(parallel_fraction_f, 0.0, 0.999)
    return 1.0 / ((1.0 - f) + (f / p_values))


def fit_parallel_fraction(p_values: list[int], speedups: list[float]) -> float:
    """
    Szacuje ulamek rownolegly 'f' z danych empirycznych (metoda najmniejszych kwadratow).
    """
    p_arr = np.array(p_values, dtype=float)
    s_arr = np.array(speedups, dtype=float)

    # Szukamy f minimalizujacego blad kwadratowy pomiedzy S_empirical a S_Amdahl
    f_candidates = np.linspace(0.50, 0.99, 500)
    best_f = 0.85
    min_sse = float("inf")

    for f in f_candidates:
        s_model = compute_amdahl_speedup(p_arr, f)
        sse = float(np.sum((s_arr - s_model) ** 2))
        if sse < min_sse:
            min_sse = sse
            best_f = float(f)

    return best_f


def plot_speedup_and_efficiency(
    benchmark_data: dict[str, Any],
    output_path: Path | None = None,
) -> Path:
    """
    Generuje dwupanelowy wykres Przyspieszenia S(p) i Efektywnosci E(p) dla OpenMP i MPI
    z uwzglednieniem linii idealnego skalowania oraz krzywej Prawa Amdahla.
    """
    if output_path is None:
        output_path = FIGURES_DIR / "speedup_efficiency.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hw = benchmark_data.get("metadata", {}).get("hardware", {})
    cpu_name = hw.get("processor", "Wielordzeniowy CPU")
    cpu_cores = hw.get("cpu_count_logical", 4)
    seed = benchmark_data.get("metadata", {}).get("seed", 42)

    openmp = benchmark_data.get("openmp_benchmark", {})
    mpi = benchmark_data.get("mpi_benchmark", {})

    omp_threads = [int(t) for t in openmp.get("thread_counts", [1, 2, 4, 8])]
    omp_speedup = [openmp["results"][str(t)]["speedup"] for t in omp_threads]
    omp_eff = [openmp["results"][str(t)]["efficiency"] * 100 for t in omp_threads]

    mpi_procs = [int(p) for p in mpi.get("process_counts", [1, 2, 4, 8])]
    mpi_speedup = [mpi["results"][str(p)]["speedup"] for p in mpi_procs]
    mpi_eff = [mpi["results"][str(p)]["efficiency"] * 100 for p in mpi_procs]

    # Teoretyczne Prawo Amdahla dla Numba/OpenMP
    f_fit = fit_parallel_fraction(omp_threads, omp_speedup)
    p_dense = np.linspace(1, max(max(omp_threads), max(mpi_procs)), 100)
    amdahl_dense = compute_amdahl_speedup(p_dense, f_fit)

    # Rysowanie wykresu
    fig, (ax_s, ax_e) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # --- Panel 1: Przyspieszenie S(p) ---
    max_p = max(max(omp_threads), max(mpi_procs))
    ax_s.plot([1, max_p], [1, max_p], "k--", alpha=0.7, label="Idealne skalowanie S(p) = p")
    ax_s.plot(p_dense, amdahl_dense, "r:", linewidth=2, label=f"Prawo Amdahla (f={f_fit*100:.1f}%)")
    ax_s.plot(omp_threads, omp_speedup, "o-", color="#1f77b4", linewidth=2.2, markersize=7, label="OpenMP / Numba CPU")
    ax_s.plot(mpi_procs, mpi_speedup, "s-", color="#2ca02c", linewidth=2.2, markersize=7, label="MPI / Local SGD")

    ax_s.set_title("Przyspieszenie S(p) w funkcji liczby wątków / procesów", fontsize=12, fontweight="bold")
    ax_s.set_xlabel("Liczba wątków / procesów p [-]", fontsize=11)
    ax_s.set_ylabel("Przyspieszenie S(p) [-]", fontsize=11)
    ax_s.set_xticks(omp_threads)
    ax_s.grid(True, linestyle="--", alpha=0.5)
    ax_s.legend(fontsize=10, loc="upper left")

    # --- Panel 2: Efektywnosc E(p) ---
    ax_e.axhline(100, color="k", linestyle="--", alpha=0.7, label="Idealna efektywność E(p) = 100%")
    ax_e.plot(omp_threads, omp_eff, "o-", color="#1f77b4", linewidth=2.2, markersize=7, label="OpenMP / Numba CPU")
    ax_e.plot(mpi_procs, mpi_eff, "s-", color="#2ca02c", linewidth=2.2, markersize=7, label="MPI / Local SGD")

    ax_e.set_title("Efektywność E(p) w funkcji liczby wątków / procesów", fontsize=12, fontweight="bold")
    ax_e.set_xlabel("Liczba wątków / procesów p [-]", fontsize=11)
    ax_e.set_ylabel("Efektywność E(p) [%]", fontsize=11)
    ax_e.set_xticks(omp_threads)
    ax_e.set_ylim(0, 110)
    ax_e.grid(True, linestyle="--", alpha=0.5)
    ax_e.legend(fontsize=10, loc="lower left")

    # Podpis ze sprzetem i ziarnem losowosci
    fig.suptitle(
        f"Ewaluacja skalowalności algorytmu Mini-Batch SGD (PRiR)\n"
        f"Sprzęt: {cpu_name} ({cpu_cores} rdzeni logicznych) | Ziarno losowości: seed={seed} | Pomiary: mediana z 5 prób",
        fontsize=11,
        y=1.03,
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"[OK] Wykres przyspieszenia i efektywnosci zapisano do: {output_path}")
    return output_path


def plot_batch_and_k_analysis(
    benchmark_data: dict[str, Any],
    output_path: Path | None = None,
) -> Path:
    """
    Generuje wykres slupkowy wplywu rozmiaru partii i wspolczynnika K na 'Czas do 74% Accuracy'.
    """
    if output_path is None:
        output_path = FIGURES_DIR / "batch_k_analysis.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hw = benchmark_data.get("metadata", {}).get("hardware", {})
    cpu_name = hw.get("processor", "Wielordzeniowy CPU")
    seed = benchmark_data.get("metadata", {}).get("seed", 42)

    sens = benchmark_data.get("batch_k_sensitivity", {})
    target_acc = sens.get("target_accuracy", 0.74)
    batch_sizes = sens.get("batch_sizes", [64, 128, 256, 512])
    k_values = sens.get("k_values", [1, 2, 4, 8, 16])
    grid_results = sens.get("grid_results", [])

    # Mapujemy wyniki na macierz (len(k_values), len(batch_sizes))
    times_matrix = np.zeros((len(k_values), len(batch_sizes)))
    for res in grid_results:
        b_idx = batch_sizes.index(res["batch_size"])
        k_idx = k_values.index(res["sync_k"])
        times_matrix[k_idx, b_idx] = res["median_time"]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

    # Rysujemy slupki grupowane
    n_groups = len(batch_sizes)
    n_bars = len(k_values)
    bar_width = 0.8 / n_bars
    x_indices = np.arange(n_groups)

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_bars))

    for k_idx, k_val in enumerate(k_values):
        offsets = x_indices - 0.4 + (k_idx + 0.5) * bar_width
        bars = ax.bar(
            offsets,
            times_matrix[k_idx, :],
            width=bar_width,
            label=f"K = {k_val}",
            color=colors[k_idx],
            edgecolor="black",
            linewidth=0.6,
        )

        # Etykiety z wartosciami na slupkach
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.2f}s",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=0,
                )

    ax.set_title(
        f"Wpływ wielkości partii (Batch Size) i parametru K na 'Czas do {target_acc * 100:.0f}% Accuracy'",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Rozmiar partii (Batch Size) [-]", fontsize=11)
    ax.set_ylabel(f"Czas osiągnięcia {target_acc * 100:.0f}% Accuracy [s]", fontsize=11)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"Batch = {b}" for b in batch_sizes], fontsize=10)
    ax.legend(title="Częstość synchronizacji (K)", fontsize=9, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    # Dodanie marginesu u gory na adnotacje
    y_max = np.max(times_matrix)
    ax.set_ylim(0, y_max * 1.15)

    fig.suptitle(
        f"Eksperyment wrażliwości Local SGD | Sprzęt: {cpu_name} | Ziarno: seed={seed} (mediana z 5 prób)",
        fontsize=10,
        y=0.98,
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"[OK] Wykres wplywu Batch Size i K zapisano do: {output_path}")
    return output_path


def plot_execution_time_comparison(
    benchmark_data: dict[str, Any],
    output_path: Path | None = None,
) -> Path:
    """
    Wykres porownujacy bezwzgledne czasy wykonania epok T(p) [s] dla OpenMP i MPI.
    """
    if output_path is None:
        output_path = FIGURES_DIR / "scaling_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    openmp = benchmark_data.get("openmp_benchmark", {})
    mpi = benchmark_data.get("mpi_benchmark", {})

    omp_threads = [int(t) for t in openmp.get("thread_counts", [1, 2, 4, 8])]
    omp_times = [openmp["results"][str(t)]["median_time"] for t in omp_threads]

    mpi_procs = [int(p) for p in mpi.get("process_counts", [1, 2, 4, 8])]
    mpi_times = [mpi["results"][str(p)]["median_time"] for p in mpi_procs]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    ax.plot(omp_threads, omp_times, "o-", color="#1f77b4", linewidth=2, label="OpenMP / Numba CPU (wątki)")
    ax.plot(mpi_procs, mpi_times, "s-", color="#2ca02c", linewidth=2, label="MPI / Local SGD (procesy)")

    for t, tm in zip(omp_threads, omp_times):
        ax.annotate(f"{tm:.3f}s", (t, tm), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    for p, tm in zip(mpi_procs, mpi_times):
        ax.annotate(f"{tm:.3f}s", (p, tm), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8)

    ax.set_title("Czas wykonania treningu T(p) w funkcji liczby jednostek obliczeniowych", fontsize=11, fontweight="bold")
    ax.set_xlabel("Liczba wątków / procesów p [-]", fontsize=10)
    ax.set_ylabel("Mediana czasu wykonania T(p) [s]", fontsize=10)
    ax.set_xticks(omp_threads)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"[OK] Wykres porownania czasow zapisano do: {output_path}")
    return output_path


def main() -> None:
    results_file = REPORTS_DIR / "benchmark_results.json"
    if not results_file.exists():
        print(f"Brak pliku wynikow {results_file}. Uruchom najpierw scripts/run_benchmarks.py!")
        return

    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    plot_speedup_and_efficiency(data)
    plot_batch_and_k_analysis(data)
    plot_execution_time_comparison(data)


if __name__ == "__main__":
    main()
