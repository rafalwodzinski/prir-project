from pathlib import Path
import subprocess
import sys
import numpy as np

from sequential_sgd import add_intercept, generate_synthetic_dataset, train_sequential_sgd
from model_mpi import split_batch_for_rank, train_mpi_sgd


def test_split_batch_for_rank_covers_entire_global_batch():
    """
    Sprawdza czy podzial globalnej partii (zarowno podzielnej jak i niepodzielnej
    przez liczbe procesow `size`) rozdziela wszystkie wiersze rozlacznie i kompletnie.
    """
    X_batch = np.arange(30).reshape(10, 3).astype(np.float32)
    y_batch = np.arange(10).astype(np.float32)
    size = 4

    chunks_x = []
    chunks_y = []
    for r in range(size):
        x_loc, y_loc = split_batch_for_rank(X_batch, y_batch, rank=r, size=size)
        chunks_x.append(x_loc)
        chunks_y.append(y_loc)

    # Dla 10 elementow i 4 procesow rozmiary lokalne powinny wynosic: 3, 3, 2, 2
    assert [ len(c) for c in chunks_y ] == [3, 3, 2, 2]
    assert np.array_equal(np.vstack(chunks_x), X_batch)
    assert np.array_equal(np.concatenate(chunks_y), y_batch)


def test_mpi_sgd_single_process_matches_baseline():
    """
    Weryfikuje dzialanie train_mpi_sgd w pojedynczym procesie wzgledem wariantu Baseline.
    """
    X, y = generate_synthetic_dataset(n_samples=512, n_features=8, seed=42, dtype=np.float32)
    X_with_bias = add_intercept(X)

    w_base, h_base = train_sequential_sgd(
        X_with_bias, y, epochs=10, batch_size=64, learning_rate=0.05, seed=99, verbose=False
    )
    w_mpi, h_mpi = train_mpi_sgd(
        X_with_bias, y, epochs=10, batch_size=64, learning_rate=0.05, seed=99, verbose=False
    )

    assert np.allclose(w_mpi, w_base, atol=1e-5)
    assert np.allclose(h_mpi["loss"], h_base["loss"], atol=1e-5)


def test_mpi_4_processes_matches_baseline_loss_curve():
    """
    DoD Issue #6:
    Uruchamia 4 procesy MPI (`mpiexec -n 4`) i sprawdza czy:
    - procesy koncza sie sukcesem (brak deadlockow, kod wyjscia 0),
    - glowny proces (rank=0) uzyskuje krzywa straty i wagi identyczne z wariantem Baseline
      przy zachowaniu globalnego rozmiaru partii.
    """
    repo_root = Path(__file__).resolve().parent.parent
    code = "\n".join([
        "import sys",
        f"sys.path.insert(0, {str(repo_root / 'src')!r})",
        "import numpy as np",
        "from mpi4py import MPI",
        "from sequential_sgd import add_intercept, generate_synthetic_dataset, train_sequential_sgd",
        "from model_mpi import train_mpi_sgd",
        "comm = MPI.COMM_WORLD",
        "rank = comm.Get_rank()",
        "X, y = generate_synthetic_dataset(n_samples=600, n_features=10, seed=42, dtype=np.float32)",
        "X_b = add_intercept(X)",
        "w_mpi, h_mpi = train_mpi_sgd(X_b, y, epochs=10, batch_size=64, learning_rate=0.05, seed=123, comm=comm, verbose=False)",
        "if rank == 0:",
        "    w_base, h_base = train_sequential_sgd(X_b, y, epochs=10, batch_size=64, learning_rate=0.05, seed=123, verbose=False)",
        "    assert np.allclose(w_mpi, w_base, atol=1e-5)",
        "    assert np.allclose(h_mpi['loss'], h_base['loss'], atol=1e-5)",
        "    print('MPI_4_PROC_OK')",
    ])

    result = subprocess.run(
        ["mpiexec", "--oversubscribe", "-n", "4", sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"Blad uruchomienia mpiexec:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert "MPI_4_PROC_OK" in result.stdout
