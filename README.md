# PRiR Project

Projekt z przedmiotu **Programowanie Równoległe i Rozproszone (PRiR)**.

---

## Struktura projektu

```text
.
├── src/                  # Kod źródłowy projektu
│   └── __init__.py
├── tests/                # Testy jednostkowe i integracyjne
│   ├── __init__.py
│   └── test_dummy.py
├── .gitignore            # Ignorowanie plików środowiska, danych i systemowych
├── pyproject.toml        # Konfiguracja projektu i pytest
├── README.md             # Dokumentacja i instrukcja uruchomienia
└── requirements.txt      # Zależności biblioteczne (Python)
```

---

## Wymagania wstępne

* Python **>= 3.10**
* *(Opcjonalnie dla `mpi4py`)* Implementacja standardu MPI w systemie:
  * **macOS:** `brew install open-mpi`
  * **Linux (Ubuntu/Debian):** `sudo apt-get install libopenmpi-dev`
  * **Windows:** [Microsoft MPI (MS-MPI)](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi) (Runtime + SDK)

---

## Konfiguracja środowiska wirtualnego (`venv`)

### 1. Klonowanie repozytorium
```bash
git clone <URL_REPOZYTORIUM>
cd prir-project
```

### 2. Utworzenie wirtualnego środowiska
W głównym katalogu projektu:

* **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  ```
* **Windows (PowerShell / Command Prompt):**
  ```powershell
  python -m venv .venv
  ```

### 3. Aktywacja środowiska
* **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```
* **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
* **Windows (Command Prompt):**
  ```cmd
  .venv\Scripts\activate.bat
  ```

Po aktywacji w prompt terminala pojawi się przedrostek `(.venv)`.

### 4. Instalacja zależności
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Uruchamianie testów

Testy jednostkowe uruchamiane są za pomocą frameworka `pytest`:

```bash
pytest
```

Dzięki konfiguracji w pliku `pyproject.toml`, katalog `src/` jest automatycznie dodawany do ścieżki Pythona (`pythonpath`), co pozwala na bezpośrednie importowanie modułów z poziomu testów.
