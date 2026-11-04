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
├── reports/                      # Zapisane raporty z eksperymentów
│   └── benchmark_results.json    # Surowe dane pomiarowe, czasy i przyspieszenia w formacie JSON
├── figures/                      # Wygenerowane wykresy naukowe
│   ├── speedup_efficiency.png    # Przyspieszenie S(p) i efektywność E(p) vs Prawo Amdahla
│   ├── scaling_comparison.png    # Porównanie czasów epok w funkcji liczby rdzeni/procesów
│   └── batch_k_analysis.png      # Czas do jakości 74% Accuracy w funkcji Batch Size i parametru K
├── tests/                        # Zestaw 33 testów jednostkowych i integracyjnych
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

Pomiary wykonano na wielordzeniowym procesorze Apple M5 Pro (15 rdzeni logicznych) oraz karcie Nvidia Tesla T4 (Google Colab).

### 1. Przyspieszenie i efektywność (Prawo Amdahla)
* Zgodnie z Prawem Amdahla:
  $$S(p) = \frac{1}{(1 - f) + \frac{f}{p}}$$
  estymowany ułamek równoległy kodu wynosi **$f \approx 87.5\%$**. 
* Dla małych partii danych narzut synchronizacji wątków/procesów dominuje nad zyskiem z podziału obliczeń. Wzrost przyspieszenia jest najbardziej widoczny przy większych partiach ($B \ge 512$).

### 2. Kompromis w Local SGD (Parametr K)
* Zwiększenie parametru rzadkiej synchronizacji $K$ ($K=4, 8, 16$) drastycznie redukuje liczbę wywołań operacji kolektywnych `Allreduce`, co w środowiskach rozproszonych o dużej latencji sieciowej przynosi znaczne oszczędności czasu.
* Zbyt duża wartość $K$ prowadzi jednak do rozbiegania się lokalnych wag modeli (*client drift*), co wymaga większej liczby epok do osiągnięcia zadanego progu dokładności. Optymalną wartością kompromisową dla badanych danych okazało się **$K = 4$**.

### 3. Akceleracja GPU CUDA
* Przeniesienie obliczeń na GPU wykazuje swoją przewagę przy dużych partiach próbek ($B \ge 1024$). 
* Utrzymanie danych rezydentnych w VRAM i przesyłanie jedynie tablicy indeksów pozwoliło ograniczyć czas transferów magistralą PCIe do ułamka milisekundy na epokę.
