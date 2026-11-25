# Regresja logistyczna z redukcją gradientów (PRiR D09)

Projekt z przedmiotu **Programowanie Równoległe i Rozproszone (PRiR)**  
**Temat 38:** Regresja logistyczna z redukcją gradientów  
**Identyfikator:** D09 | **Zespół:** 1–2 osoby  

---

## Spis treści
1. [Opis projektu i cel badawczy](#opis-projektu-i-cel-badawczy)
2. [Struktura repozytorium](#struktura-repozytorium)
3. [Wymagania wstępne i instalacja](#wymagania-wstępne-i-instalacja)
4. [Architektura i warianty obliczeniowe](#architektura-i-warianty-obliczeniowe)
   * [1. Baseline sekwencyjny (NumPy)](#1-baseline-sekwencyjny-numpy)
   * [2. Wielowątkowość CPU / Ekwiwalent OpenMP (Numba JIT)](#2-wielowatkowosc-cpu--ekwiwalent-openmp-numba-jit)
   * [3. Klaster rozproszony MPI (mpi4py + Allreduce)](#3-klaster-rozproszony-mpi-mpi4py--allreduce)
   * [4. Local SGD z rzadką synchronizacją wag (parametr K)](#4-local-sgd-z-rzadka-synchronizacja-wag-parametr-k)
   * [5. Akceleracja GPU CUDA (Numba CUDA + Google Colab)](#5-akceleracja-gpu-cuda-numba-cuda--google-colab)
5. [Instrukcja uruchomienia (CLI)](#instrukcja-uruchomienia-cli)
   * [Uruchamianie przez uniwersalny skrypt main.py](#uruchamianie-przez-uniwersalny-skrypt-mainpy)
   * [Uruchamianie bezpośrednich modułów i klastra MPI](#uruchamianie-bezposrednich-modulow-i-klastra-mpi)
   * [Benchmarki wydajnościowe i generowanie wykresów](#benchmarki-wydajnosciowe-i-generowanie-wykresow)
   * [Uruchamianie testów jednostkowych (pytest)](#uruchamianie-testow-jednostkowych-pytest)
6. [Zbiór danych i stabilność numeryczna](#zbior-danych-i-stabilnosc-numeryczna)
7. [Weryfikacja numeryczna gradientu (Gradient Checking)](#weryfikacja-numeryczna-gradientu-gradient-checking)
8. [Wyniki eksperymentów i wnioski](#wyniki-eksperymentow-i-wnioski)

---

## Opis projektu i cel badawczy

Projekt implementuje algorytm optymalizacji **Mini-Batch Stochastic Gradient Descent (SGD)** dla binarnej regresji logistycznej z funkcją straty Binary Cross-Entropy (BCE). 

W każdej iteracji algorytmu:
1. Obliczane są predykcje prawdopodobieństw $\hat{y} = \sigma(X W)$ przy użyciu numerycznie stabilnej sigmoidy.
2. Wyznaczana jest funkcja straty oraz wektorowy gradient $\nabla_W L = \frac{1}{B} X^T (\hat{y} - y)$.
3. Gradienty częściowe z poszczególnych wątków, procesów lub bloków GPU są redukowane przed wykonaniem wspólnej aktualizacji wag:
   $$W \leftarrow W - \alpha \cdot \nabla_W L$$

Głównym celem badawczym jest **ocena przyspieszenia $S(p)$ i efektywności $E(p)$ w odniesieniu do Prawa Amdahla** oraz zbadanie wpływu wielkości partii (batch size) i częstotliwości synchronizacji (parametr $K$) na metrykę **Time-to-Quality** (czas osiągnięcia zadanego progu dokładności).

---

## Struktura repozytorium

```text
.
├── src/                          # Kod źródłowy algorytmów uczenia i narzędzi
│   ├── main.py                   # Główny interfejs CLI (argparse, zapis modelu i historii, Time-to-Quality)
│   ├── sequential_sgd.py         # Wariant sekwencyjny (Baseline NumPy)
│   ├── numba_sgd.py              # Wariant wielowątkowy CPU (Numba @njit(parallel=True), prange, Map-Reduce)
│   ├── model_mpi.py              # Wariant rozproszony MPI (Allreduce) oraz Local SGD (parametr K)
│   ├── cuda_sgd.py               # Wariant GPU CUDA (dane w VRAM, stochastyczne indeksowanie)
│   ├── data_loader.py            # Wczytywanie danych CS:GO, kodowanie cech, podział i standaryzacja
│   └── math_utils.py             # Stabilna sigmoida, BCE loss z zabezpieczeniem eps, wektorowy gradient
├── scripts/                      # Skrypty pomiarowe i wizualizacyjne
│   ├── run_benchmarks.py         # Pomiary skalowalności OpenMP, MPI (mpiexec) oraz czułości Batch-K
│   ├── plot_results.py           # Generowanie publikacyjnych wykresów PNG (Amdahl, scaling, heatmaps)
│   └── verify_real_data.py       # Szybka weryfikacja poprawności wczytywania zbioru CS:GO
├── notebooks/                    # Notatniki demonstracyjne dla chmury
│   └── cuda_colab.ipynb          # Notatnik Google Colab z akceleratorem Nvidia Tesla T4
├── reports/                      # Zapisane raporty z eksperymentów (każdy zawiera metadane sprzętu)
│   ├── benchmark_results.json    # Wyniki skalowania CPU (Mac), Batch-K oraz skalowania rozmiaru danych (Colab T4)
│   ├── cuda_results.json         # Podsumowanie porównania CUDA vs Sequential na CS:GO (Colab T4)
│   ├── colab_cuda_history.json   # Surowa historia epok dla wariantu CUDA z sesji Google Colab
│   └── colab_seq_history.json    # Surowa historia epok dla wariantu Sequential z tej samej sesji Google Colab
├── figures/                      # Wygenerowane wykresy naukowe
│   ├── speedup_efficiency.png    # Przyspieszenie S(p) i efektywność E(p) (vs T_seq i vs T₁) oraz Prawo Amdahla
│   ├── scaling_comparison.png    # Porównanie czasów epok w funkcji liczby rdzeni/procesów vs baseline T_seq
│   ├── batch_k_analysis.png      # Czas do jakości 74% Accuracy w funkcji Batch Size i parametru K
│   ├── data_size_scaling.png     # Skalowanie względem rozmiaru danych (100k–5M próbek): Seq, Numba, MPI, CUDA
│   └── cuda_comparison.png       # Porównanie czasów epok i zgodności Accuracy: CUDA vs Sequential (Colab T4)
├── tests/                        # Zestaw 36 testów jednostkowych i integracyjnych
│   ├── conftest.py               # Automatyczna izolacja symulatora CUDASIM na macOS/laptopach bez NVIDIA
│   ├── test_cli.py               # Testy argumentów CLI, serializacji modeli .npy/.npz i historii JSON
│   ├── test_cuda.py              # Testy poprawności kerneli CUDA i zbieżności SGD na GPU
│   ├── test_local_sgd.py         # Testy wariantu Local SGD z parametrem K
│   ├── test_mpi.py               # Testy partycjonowania partii i wieloprocesowego mpiexec
│   ├── test_numba.py             # Testy spójności wag Numby z wariantem Baseline
│   ├── test_gradient.py          # Numeryczny gradient check metodą różnic skończonych centralnych
│   ├── test_data.py              # Testy braku wycieku danych (data leakage) i formatu C-contiguous
│   ├── test_math.py              # Testy stabilności numerycznej exp/log w skrajnych wartościach
│   └── test_benchmarks.py        # Testy funkcji pomiarowych i własności Prawa Amdahla
├── data/                         # Katalog na dane wejściowe (ignorowany w git)
│   └── csgo_round_snapshots.csv  # Dataset CS:GO (~122k rund, opcjonalny - fallback na syntetyk)
├── pyproject.toml                # Konfiguracja środowiska i pytest (pythonpath = ["src"])
├── requirements.txt              # Zależności biblioteczne (numpy, pandas, scikit-learn, numba, mpi4py, matplotlib, pytest)
└── README.md                     # Niniejsza dokumentacja
```

---

## Wymagania wstępne i instalacja

* **Python:** wersja `>= 3.10`
* **Implementacja MPI w systemie operacyjnym:**
  * **macOS:** `brew install open-mpi`
  * **Linux (Ubuntu/Debian):** `sudo apt-get install libopenmpi-dev`
  * **Windows:** [Microsoft MPI (MS-MPI)](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi) (Runtime + SDK)

### Krok po kroku:

```bash
# 1. Klonowanie repozytorium
git clone https://github.com/rafalwodzinski/prir-project.git
cd prir-project

# 2. Utworzenie i aktywacja środowiska wirtualnego
python3 -m venv .venv

# macOS / Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# 3. Instalacja zależności
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Architektura i warianty obliczeniowe

W projekcie zaimplementowano 5 wariantów optymalizacji, gwarantujących identyczną zbieżność matematyczną przy różnych paradygmatach zrównoleglenia:

### 1. Baseline sekwencyjny (NumPy)
* **Plik:** [`src/sequential_sgd.py`](file:///Users/rafalwodzinski/Studia/PRiR/Projekt/prir-project/src/sequential_sgd.py)
* **Mechanika:** Klasyczny Mini-Batch SGD. Tablica cech jest tasowana w każdej epoce (`rng.permutation`), dzielona na równe partię o wielkości $B$, a gradient liczony jest wektorowo: `(X.T @ (y_pred - y_true)) / batch_size`. Stanowi punkt odniesienia (złoty standard) dla weryfikacji poprawności pozostałych wariantów.

### 2. Wielowątkowość CPU / Ekwiwalent OpenMP (Numba JIT)
* **Plik:** [`src/numba_sgd.py`](file:///Users/rafalwodzinski/Studia/PRiR/Projekt/prir-project/src/numba_sgd.py)
* **Uzasadnienie formalne OpenMP w Pythonie:** W środowisku Python standardowym odpowiednikiem dyrektyw `#pragma omp parallel for` z biblioteki OpenMP jest kompilator JIT biblioteki **Numba** z dekoratorem `@njit(parallel=True)` i pętlą `prange()`, oparty pod spodem o backend wielowątkowości OpenMP/TBB.
* **Ochrona przed wyścigami (Race Conditions):** Zastosowano wzorzec **Map-Reduce**:
  * **Faza Map:** Każdy wątek sprzętowy pobiera swój identyfikator `tid = get_thread_id()` i akumuluje częściowy gradient próbek we własnym prywatnym buforze `local_grads[tid, j]`. Wyklucza to jednoczesny zapis do tej samej komórki pamięci w rejestrach CPU i eliminuje zjawisko *False Sharing*.
  * **Faza Reduce:** Po zakończeniu pętli równoległej następuje bezpieczna redukcja (zsumowanie wierszy tablicy wątków) i pojedyncza aktualizacja wag.

### 3. Klaster rozproszony MPI (mpi4py + Allreduce)
* **Plik:** [`src/model_mpi.py`](file:///Users/rafalwodzinski/Studia/PRiR/Projekt/prir-project/src/model_mpi.py)
* **Mechanika:** Synchroniczny *Data-Parallel Distributed SGD*. 
  * Globalny mini-batch $B_{\text{global}}$ jest rozdzielany na $P$ rozłącznych podzbiorów lokalnych $B_{\text{local}} \approx B_{\text{global}} / P$ za pomocą funkcji `split_batch_for_rank`.
  * Każdy węzeł/proces oblicza lokalną sumę błędów na swoim wycinku danych: $g_{\text{local}} = X_{\text{local}}^T (\hat{y}_{\text{local}} - y_{\text{local}})$.
  * Wywołanie kolektywne `comm.Allreduce(local_grad, global_grad, op=MPI.SUM)` synchronizuje wektor gradientu przez sieć, po czym każdy proces wykonuje identyczny krok aktualizacji wag. Wyniki są matematycznie w 100% tożsame z wariantem sekwencyjnym.

### 4. Local SGD z rzadką synchronizacją wag (parametr K)
* **Plik:** [`src/model_mpi.py`](file:///Users/rafalwodzinski/Studia/PRiR/Projekt/prir-project/src/model_mpi.py) (`train_local_sgd_mpi`)
* **Zakres 2-osobowy:** Wariant redukujący wąskie gardło komunikacji sieciowej. 
  * Węzły nie komunikują się przy każdym mini-batchu. Aktualizują swoje wagi lokalnie przez $K$ kroków.
  * Dopiero co $K$ iteracji następuje operacja uśredniania wag modeli: `comm.Allreduce(weights, global_weights_sum, op=MPI.SUM)` i przypisanie $W \leftarrow \frac{1}{P} \sum W_p$.
  * Pozwala na znaczne przyspieszenie w sieciach o wysokiej latencji kosztem nieznacznego rozbiegania się lokalnych modeli.

### 5. Akceleracja GPU CUDA (Numba CUDA + Google Colab)
* **Plik:** [`src/cuda_sgd.py`](file:///Users/rafalwodzinski/Studia/PRiR/Projekt/prir-project/src/cuda_sgd.py) oraz notatnik [`notebooks/cuda_colab.ipynb`](file:///Users/rafalwodzinski/Studia/PRiR/Projekt/prir-project/notebooks/cuda_colab.ipynb)
* **Optymalizacja transferów PCIe:** Zbiór danych cech $X$ i etykiet $y$ przesyłany jest do pamięci VRAM karty graficznej **tylko raz na początku uczenia**.
* **Stochastyczne indeksowanie:** Zamiast kosztownego tasowania i ponownego kopiowania 50 MB danych w każdej epoce, CPU przesyła jedynie lekką tablicę przetasowanych indeksów `d_indices` (typ `int32`, poniżej 400 KB). Wątki GPU odczytują wiersze pośrednio przez `row = d_indices[batch_start + i]`.
* **Praca na maszynach bez kart NVIDIA:** Dzięki automatycznej konfiguracji w `tests/conftest.py`, na komputerach bez fizycznego GPU (np. MacBook Apple Silicon lub laptopy ze zintegrowaną grafiką) moduł bezbłędnie wykonuje testy i symulację w trybie programowym **CUDASIM** (`NUMBA_ENABLE_CUDASIM=1`). Pomiary na fizycznym akceleratorze Nvidia Tesla T4 wykonano w dołączonym notatniku Google Colab.

---

## Instrukcja uruchomienia (CLI)

### Uruchamianie przez uniwersalny skrypt main.py

Projekt posiada zunifikowany interfejs wiersza poleceń [`src/main.py`](file:///Users/rafalwodzinski/Studia/PRiR/Projekt/prir-project/src/main.py) z pełną obsługą flag, automatycznym zapisem wag wyuczonego modelu (`.npy` lub `.npz`), historii uczenia (`.json`) oraz wyznaczaniem metryki *Time-to-Quality*:

```bash
# Wariant 1: Trening sekwencyjny (Baseline) z zapisem modelu i historii
python src/main.py --mode seq --epochs 20 --batch-size 256 --lr 0.01 --target-acc 0.75 --save-model models/weights.npy --save-history reports/history.json

# Wariant 2: Wielowatkowy CPU (Numba / OpenMP)
python src/main.py --mode numba --epochs 20 --batch-size 256 --lr 0.01

# Wariant 3: Klaster rozproszony MPI (np. 4 procesy robocze)
mpiexec -n 4 python src/main.py --mode mpi --epochs 20 --batch-size 256 --lr 0.01

# Wariant 4: Local SGD z rzadka synchronizacja co K=4 kroki
mpiexec -n 4 python src/main.py --mode mpi --k 4 --epochs 20 --batch-size 256

# Wariant 5: Akceleracja GPU CUDA
python src/main.py --mode cuda --epochs 20 --batch-size 256 --lr 0.01
```

#### Dostępne parametry CLI:
| Flaga | Typ | Domyślnie | Opis |
|---|---|---|---|
| `--mode` | str | `seq` | Wybór wariantu: `seq`, `numba`, `mpi`, `cuda` |
| `--lr` | float | `0.01` | Współczynnik uczenia (*learning rate*) |
| `--batch-size`| int | `256` | Wielkość globalnej partii (*mini-batch size*) |
| `--epochs` | int | `20` | Liczba epok uczenia |
| `--seed` | int | `42` | Ziarno generatora losowego (determinizm) |
| `--data` | str | `data/csgo_round_snapshots.csv` | Ścieżka do pliku z danymi CSV |
| `--target-acc`| float | `0.75` | Docelowy próg dokładności dla metryki *Time-to-Quality* |
| `--save-model`| str | `None` | Ścieżka docelowa zapisu wag modelu (`.npy` lub `.npz`) |
| `--save-history`| str| `None` | Ścieżka docelowa zapisu historii metryk (`.json`) |
| `--k` | int | `4` | Częstotliwość synchronizacji wag w wariancie Local SGD |
| `--quiet` | flaga | `False` | Wyciszenie szczegółowych logów poszczególnych epok |

---

### Uruchamianie bezpośrednich modułów i klastra MPI

Każdy moduł można również uruchomić niezależnie jako samodzielny skrypt:

```bash
# Uruchomienie modelu sekwencyjnego
python src/sequential_sgd.py

# Uruchomienie modelu wielowatkowego Numba
python src/numba_sgd.py

# Uruchomienie modelu rozproszonego MPI na 4 procesach
mpiexec -n 4 python src/model_mpi.py

# Uruchomienie modelu CUDA (lub CUDASIM)
python src/cuda_sgd.py
```

---

### Benchmarki wydajnościowe i generowanie wykresów

Projekt zawiera automatyczny potok badań wydajnościowych:

```bash
# 1. Uruchomienie pelnego zestawu testow wydajnosciowych (zapis do reports/benchmark_results.json)
python scripts/run_benchmarks.py

# 2. Wygenerowanie wykresow naukowych (zapis do figures/*.png)
python scripts/plot_results.py
```

Wygenerowane wykresy trafiają do katalogu `figures/`:
* `figures/speedup_efficiency.png` – krzywe przyspieszenia i efektywności względem Prawa Amdahla.
* `figures/scaling_comparison.png` – czasy wykonania w zależności od liczby rdzeni/procesów.
* `figures/batch_k_analysis.png` – analiza wrażliwości metryki *Time-to-Quality* na wielkość partii i parametr $K$.

---

### Uruchamianie testów jednostkowych (pytest)

Zestaw 33 testów jednostkowych i integracyjnych weryfikuje wszystkie aspekty projektu:

```bash
# Uruchomienie wszystkich testow
pytest -v

# Uruchomienie weryfikacji numerycznej gradientu
pytest tests/test_gradient.py -v

# Uruchomienie testow klastra MPI
pytest tests/test_mpi.py -v

# Uruchomienie testow CLI
pytest tests/test_cli.py -v
```

---

## Zbiór danych i stabilność numeryczna

* **Dane:** Domyślnie wykorzystywany jest zbiór *CS:GO Round Snapshots* (`data/csgo_round_snapshots.csv`), zawierający 122 410 rund meczowych i 104 cechy (po zakodowaniu One-Hot zmiennych kategorycznych map oraz binarnej zmiennej `bomb_planted`).
* **Fallback syntetyczny:** Jeśli plik CSV nie znajduje się na dysku, skrypty automatycznie generują deterministyczny zbiór syntetyczny (`generate_synthetic_dataset`), co pozwala na natychmiastowe uruchomienie projektu w dowolnym środowisku.
* **Brak wycieku danych (Data Leakage):** Zgodnie ze specyfikacją zadania, skalowanie cech (`StandardScaler`) dopasowywane jest **wyłącznie na zbiorze treningowym** (`fit_transform` na train, tylko `transform` na test/val).
* **Stabilność numeryczna:**
  * Funkcja sigmoidalna $\sigma(z) = \frac{1}{1 + e^{-z}}$ zabezpieczona jest przed przepełnieniem (overflow/underflow) poprzez obcięcie argumentu `z = np.clip(z, -85.0, 85.0)`.
  * Funkcja straty BCE obcina predykcje do zakresu $[\epsilon, 1-\epsilon]$, gdzie dla typu `float32` próg wynosi bezpieczne $\epsilon = 10^{-7}$ (zapobiega to zaokrągleniu $1.0 - 10^{-15}$ z powrotem do $1.0$ i powstawaniu wartości `NaN`).

---

## Weryfikacja numeryczna gradientu (Gradient Checking)

W module [`tests/test_gradient.py`](file:///Users/rafalwodzinski/Studia/PRiR/Projekt/prir-project/tests/test_gradient.py) zaimplementowano numeryczny dowód poprawności wektorowego gradientu analitycznego metodą **różnic skończonych centralnych**:

$$\frac{\partial L}{\partial W_j} \approx \frac{L(W + \epsilon \cdot e_j) - L(W - \epsilon \cdot e_j)}{2\epsilon}$$

Dzięki rozwinięciu w szereg Taylora błąd metody centralnej maleje kwadratowo $O(\epsilon^2)$ (w przeciwieństwie do różnic jednostronnych $O(\epsilon)$). Dla $\epsilon = 10^{-5}$ uzyskano pełną zgodność gradientu analitycznego z numerycznym:
$$\text{np.allclose}(grad\_analityczny, grad\_numeryczny, atol=1e-5) \implies \mathbf{PASSED}$$

---

## Wyniki eksperymentów i wnioski

### Środowiska pomiarowe (tabela: wynik → środowisko)

> **Ważne zastrzeżenie metodologiczne:** Pomiary CPU (OpenMP/Numba, MPI, Batch–K) wykonano lokalnie na Apple Silicon, natomiast pomiary z akceleratorem GPU CUDA wykonano w Google Colab na karcie NVIDIA Tesla T4. **Bezwzględnych czasów wykonania z Apple M5 Pro i Google Colab nie należy bezpośrednio ze sobą porównywać** — w każdym środowisku warianty równoległe są odnoszone do baseline'u sekwencyjnego (`T_seq`) zmierzonego na **tej samej maszynie**.

| Plik wyników / wykres | Zakres eksperymentu | Środowisko sprzętowe i programowe | Data pomiaru |
|---|---|---|---|
| `reports/benchmark_results.json` (`sequential_baseline`, `openmp_benchmark`, `mpi_benchmark`, `batch_k_sensitivity`)<br>`figures/speedup_efficiency.png`<br>`figures/scaling_comparison.png`<br>`figures/batch_k_analysis.png` | CS:GO (97 928 próbek × 104 cechy): baseline `T_seq`, skalowanie OpenMP/Numba (1–8T), MPI (`mpiexec`, 1–8P, $K=4$), analiza Batch–K do 74% Accuracy | **Apple M5 Pro** (`arm`, 15 rdzeni logicznych), macOS (Darwin 27.0.0), Python 3.13.13, NumPy 2.5.4, Numba 0.68.0, OpenMPI | 2026-11-25 |
| `reports/benchmark_results.json` (`data_size_scaling`)<br>`figures/data_size_scaling.png` | Skalowanie rozmiaru danych syntetycznych (100k, 500k, 1M, 5M próbek × 101 cech, `batch=4096`): Sequential, Numba (8T), MPI (4P `mpiexec`), **CUDA** | **Google Colab** (Linux x86_64, Intel Xeon vCPU + **NVIDIA Tesla T4** 16 GB VRAM), Python 3.13, Numba CUDA, OpenMPI | 2026-11-12 |
| `reports/cuda_results.json`<br>`reports/colab_cuda_history.json`<br>`reports/colab_seq_history.json`<br>`figures/cuda_comparison.png` | Porównanie 20 epok CUDA vs Sequential na zbiorze CS:GO (`batch=512`, `lr=0.003`): czas epoki oraz zgodność krzywych Accuracy i Loss | **Google Colab** (Linux x86_64, Intel Xeon vCPU + **NVIDIA Tesla T4** 16 GB VRAM), Python 3.13, Numba CUDA | 2026-11-04 |

Szczegółowe metadane sprzętowe (CPU, model GPU, wersje bibliotek oraz data) zapisywane są automatycznie w sekcji `"metadata"` / `"environment"` każdego pliku wynikowego JSON w `reports/`.

### Metodologia pomiaru czasu (porównywalność wariantów)

Wszystkie warianty CPU (`Sequential`, `OpenMP / Numba`, `MPI`) w `scripts/run_benchmarks.py` raportują **tę samą metrykę**: medianę z 5 pomiarów (po 1 przebiegu rozgrzewkowym) **sumy `epoch_times`**, obejmującą wyłącznie matematykę epok SGD i synchronizację — **bez** wczytywania CSV, bez jednorazowej kompilacji JIT i bez ewaluacji `predict_proba`:

| Wariant | Co obejmuje zegar `epoch_times` | Co jest wykluczone z pomiaru |
|---|---|---|
| **Sequential (NumPy)** | Tasowanie indeksów + pętla mini-batch SGD (`X @ w`, `X.T @ err` przez BLAS) + aktualizacja wag | Wczytanie/skalowanie danych, ewaluacja `predict_proba` / `BCE` |
| **OpenMP / Numba** | Tasowanie indeksów + równoległy kernel `compute_gradient_numba` (`prange`, Map-Reduce) + aktualizacja wag | Kompilacja JIT (wykonana w rozgrzewce), I/O, ewaluacja |
| **MPI (`mpiexec`)** | Tasowanie indeksów + podział batcha + lokalne SGD (NumPy) + `comm.Allreduce` co $K$ kroków (mierzone na `rank=0`) | Uruchomienie `mpiexec`, I/O danych, ewaluacja |
| **CUDA (GPU)** | Przesłanie lekkiej tablicy indeksów `int32` + 3 kernele GPU, otoczone `cuda.synchronize()` | Jednorazowy transfer `X, y` do VRAM, 1. epoka (JIT), `copy_to_host` wag i ewaluacja |

> **Dlaczego $T_1$ Numby (`0.150 s`) jest wyższe od $T_1$ MPI (`0.0385 s`) mimo identycznego zakresu pomiaru?**  
> Na 1 wątku kernel Numby wykonuje jawne zagnieżdżone pętle skalarne po wierszach i cechach, natomiast `Sequential` i `MPI` (dla $p=1$) korzystają ze zwektoryzowanych operacji macierzowych BLAS/Accelerate w NumPy (`X @ weights`, `X.T @ error`). Dzięki ujednoliceniu zakresu oba czasy są w pełni porównywalne na jednym wykresie (`figures/scaling_comparison.png`).

### 1. Przyspieszenie i efektywność (CS:GO vs `T_seq` i Prawo Amdahla)
* Na zbiorze CS:GO (~98 tys. próbek × 104 cechy, `batch=512`) bazowa wersja sekwencyjna NumPy wykonuje 3 epoki w **`T_seq = 0.0346 s`**.
* Dla tak niewielkiego kosztu obliczeniowego na krok narzut uruchamiania wątków (`prange`) i komunikacji międzyprocesowej (`mpiexec`) przewyższa zysk ze zrównoleglenia: względem `T_seq` OpenMP/Numba osiąga `0.20x–0.24x`, a MPI `0.52x–0.90x`. Pokazuje to wprost ograniczenia Prawa Amdahla dla problemów o niskiej intensywności obliczeniowej.

### 2. Kompromis w Local SGD (Parametr K i rozmiar partii)
* Na CS:GO regresja logistyczna osiąga sufit ~75% Accuracy, dlatego w eksperymencie czułości zbadano czas do osiągnięcia **74% Accuracy** (`lr=0.003`).
* Najkrótszy czas daje `batch=128, K=16` (~26.3 ms, 1 epoka); większe wartości $K$ redukują koszt uśredniania wag o kilka–kilkanaście procent przy zachowaniu praktycznie identycznej dokładności.

### 3. Akceleracja GPU CUDA i skalowanie rozmiaru danych
* Na małym zbiorze CS:GO (`batch=512`, `cuda_comparison.png`) GPU osiąga identyczną dokładność i stratę co CPU (`75.13%`, Loss `0.4839`), ale jest wolniejsze (`0.40x`) przez narzut uruchamiania małych kerneli.
* W eksperymencie skalowania rozmiaru danych (`data_size_scaling.png`, `batch=4096`, Tesla T4) **próg opłacalności GPU wypada między 100k a 500k próbek**: przy 100k CUDA ma `0.96x`, natomiast od **500k do 5M próbek wyraźnie wygrywa z wersją sekwencyjną, osiągając `1.58x–1.69x` przyspieszenia**.
* Utrzymanie danych rezydentnych w VRAM i przesyłanie jedynie tablicy indeksów `int32` eliminuje wąskie gardło magistrali PCIe w pętli epok.
