import os
import platform

# Automatyczne wlaczenie symulatora CUDASIM na maszynach bez fizycznej karty NVIDIA (np. macOS Apple Silicon,
# laptopy ze zintegrowana grafika), aby pytest przechodzil lokalnie bez instalacji sterownikow CUDA.
# Zmienna srodowiskowa musi byc ustawiona przed pierwszym zaimportowaniem biblioteki Numba.
if "NUMBA_ENABLE_CUDASIM" not in os.environ:
    if platform.system() == "Darwin":
        os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
    else:
        try:
            import ctypes
            lib_name = "nvcuda.dll" if platform.system() == "Windows" else "libcuda.so.1"
            ctypes.CDLL(lib_name)
        except Exception:
            os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

