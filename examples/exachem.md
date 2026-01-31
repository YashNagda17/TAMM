# ExaChem Setup Guide

ExaChem is a quantum chemistry application that uses TAMM as its tensor algebra backend. For coupled-cluster calculations (CCSD, CCSD(T)), **90%+ of compute time is spent in TAMM tensor contractions**.

## Repository

- **GitHub**: https://github.com/ExaChem/exachem
- **Documentation**: https://exachem.github.io/exachem/

---

## Environment Setup for CUDA Builds

```bash
# Set CUDA environment (required for TAMM/ExaChem CUDA builds)
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Disable conda (can interfere with system compilers)
conda deactivate 2>/dev/null
unset CONDA_EXE CONDA_PREFIX CONDA_PYTHON_EXE CONDA_DEFAULT_ENV CONDA_SHLVL

# Verify CUDA
nvcc --version
```



## 1. Prerequisites

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y \
  build-essential \
  gfortran \
  cmake \
  openmpi-bin \
  libopenmpi-dev \
  libblas-dev \
  liblapack-dev \
  git \
  python3 \
  python3-pip

# For CUDA support (optional)
# Install CUDA toolkit from NVIDIA
```

---

## 2. Build Approaches

There are two ways to build ExaChem:

| Approach | Use Case | Rebuild Time |
|----------|----------|--------------|
| **Embedded TAMM** | Simple setup, no TAMM modifications | Full rebuild required |
| **External TAMM** | TAMM development/modifications | Only rebuild TAMM + relink |

**Recommended**: Use **External TAMM** if you plan to modify TAMM's code (e.g., `gpu_blas.cpp`).

---

## 2.1 Build with Embedded TAMM (Simple)

Use this if you don't need to modify TAMM.

```bash
# Set paths
export EXACHEM_ROOT=$HOME/exachem
export EXACHEM_INSTALL=$HOME/exachem_install

# Clone with submodules (includes TAMM)
git clone --recursive https://github.com/ExaChem/exachem.git $EXACHEM_ROOT
cd $EXACHEM_ROOT

# Create build directory
mkdir build && cd build

# Configure (CPU only)
CC=gcc CXX=g++ FC=gfortran cmake \
  -DCMAKE_INSTALL_PREFIX=$EXACHEM_INSTALL \
  -DALLOW_CONDA=ON \
  ..

# Configure (with CUDA - uncomment and adjust GPU_ARCH)
# Other modules = SCF;CD;CC
CC=gcc CXX=g++ FC=gfortran cmake \
  -DCMAKE_INSTALL_PREFIX=$EXACHEM_INSTALL \
  -DTAMM_ENABLE_CUDA=ON \
  -DGPU_ARCH=61 \
  -DALLOW_CONDA=ON \
  -DMODULES="CC" \
  ..

make -j$(nproc)
make install
```

---

## 2.2 Build with External TAMM (For TAMM Development)

Use this approach when you need to modify TAMM code. TAMM is built separately, and ExaChem links to it.

### Step 1: Build TAMM Standalone

```bash
# Clone TAMM in current directory
git clone https://github.com/NWChemEx-Project/TAMM.git
cd TAMM

# Make your modifications (e.g., gpu_blas.cpp profiling)
# vim src/tamm/kernels/gpu_blas.cpp

# Create build directory
mkdir build && cd build

# Configure with CUDA (install to ../install)
CC=gcc CXX=g++ FC=gfortran cmake \
  -DCMAKE_INSTALL_PREFIX=$(pwd)/../install \
  -DTAMM_ENABLE_CUDA=ON \
  -DGPU_ARCH=61 \
  -DALLOW_CONDA=ON \
  ..

make -j$(nproc)
make install
```

### Step 2: Build ExaChem Linking to External TAMM

```bash
# Go back to parent directory
cd ../..

# Clone ExaChem (WITHOUT --recursive to skip bundled TAMM)
git clone https://github.com/ExaChem/exachem.git
cd exachem

# Create build directory
mkdir build && cd build

# Configure with external TAMM (adjust path to your TAMM install)
TAMM_INSTALL=/home/yash/Desktop/Applications/TAMM/install

CC=gcc CXX=g++ FC=gfortran cmake \
  -DCMAKE_INSTALL_PREFIX=$(pwd)/../install \
  -DTAMM_ENABLE_CUDA=ON \
  -DGPU_ARCH=61 \
  -DCMAKE_CUDA_ARCHITECTURES=61\
  -DALLOW_CONDA=ON \
  -DMODULES="CC" \
  ..

make -j$(nproc)
make install
```

**Important**: The `CMSB_DEBUG_CMAKE=ON` flag enables the build system to find pre-installed packages instead of building them from source.

### Step 2b: Fix CMake External Dependency Errors

After running cmake, if you see errors like:
```
CMake Error: The dependency target "BLAS_External" of target "GlobalArrays_External" does not exist.
CMake Error: The dependency target "BLAS_External" of target "GauXC_External" does not exist.
```

This occurs because BLIS/LAPACK are found as pre-installed libraries (not built from source), so the `BLAS_External` and `LAPACK_External` targets don't exist. Fix by editing the CMake files:

#### Fix 1: BuildGlobalArrays.cmake

**File location (after cmake configure):**
- TAMM: `build/_deps/cmakebuild-src/cmake/build_external/BuildGlobalArrays.cmake`
- ExaChem: `build/_deps/cmakebuild-src/cmake/build_external/BuildGlobalArrays.cmake`

**Find this code (around line 165-175):**
```cmake
# Establish the dependencies
if(${LINALG_VENDOR} STREQUAL "BLIS" OR ${LINALG_VENDOR} STREQUAL "IBMESSL" OR ${LINALG_VENDOR} STREQUAL "OpenBLAS")
    if(${LINALG_VENDOR} STREQUAL "BLIS" OR ${LINALG_VENDOR} STREQUAL "OpenBLAS")
        add_dependencies(GlobalArrays_External BLAS_External LAPACK_External)
    elseif(${LINALG_VENDOR} STREQUAL "IBMESSL")
        add_dependencies(GlobalArrays_External LAPACK_External)
    endif()
    if(USE_SCALAPACK)
        add_dependencies(GlobalArrays_External ScaLAPACK_External)
    endif()
endif()
```

**Replace with:**
```cmake
# Establish the dependencies (only if targets exist - they won't if libraries were found)
if(${LINALG_VENDOR} STREQUAL "BLIS" OR ${LINALG_VENDOR} STREQUAL "IBMESSL" OR ${LINALG_VENDOR} STREQUAL "OpenBLAS")
    if(${LINALG_VENDOR} STREQUAL "BLIS" OR ${LINALG_VENDOR} STREQUAL "OpenBLAS")
        if(TARGET BLAS_External)
            add_dependencies(GlobalArrays_External BLAS_External)
        endif()
        if(TARGET LAPACK_External)
            add_dependencies(GlobalArrays_External LAPACK_External)
        endif()
    elseif(${LINALG_VENDOR} STREQUAL "IBMESSL")
        if(TARGET LAPACK_External)
            add_dependencies(GlobalArrays_External LAPACK_External)
        endif()
    endif()
    if(USE_SCALAPACK)
        if(TARGET ScaLAPACK_External)
            add_dependencies(GlobalArrays_External ScaLAPACK_External)
        endif()
    endif()
endif()
```

#### Fix 2: BuildGauXC.cmake (ExaChem only)

**File location:**
`build/_deps/cmakebuild-src/cmake/build_external/BuildGauXC.cmake`

**Find this code (around line 127-136):**
```cmake
if(${LINALG_VENDOR} STREQUAL "BLIS" OR ${LINALG_VENDOR} STREQUAL "IBMESSL" OR ${LINALG_VENDOR} STREQUAL "OpenBLAS")
    if(${LINALG_VENDOR} STREQUAL "BLIS" OR ${LINALG_VENDOR} STREQUAL "OpenBLAS")
        add_dependencies(GauXC_External BLAS_External LAPACK_External)
    elseif(${LINALG_VENDOR} STREQUAL "IBMESSL")
        add_dependencies(GauXC_External LAPACK_External)
    endif()
    if(USE_SCALAPACK)
        add_dependencies(GauXC_External ScaLAPACK_External)
    endif()
endif()
```

**Replace with:**
```cmake
if(${LINALG_VENDOR} STREQUAL "BLIS" OR ${LINALG_VENDOR} STREQUAL "IBMESSL" OR ${LINALG_VENDOR} STREQUAL "OpenBLAS")
    if(${LINALG_VENDOR} STREQUAL "BLIS" OR ${LINALG_VENDOR} STREQUAL "OpenBLAS")
        if(TARGET BLAS_External)
            add_dependencies(GauXC_External BLAS_External)
        endif()
        if(TARGET LAPACK_External)
            add_dependencies(GauXC_External LAPACK_External)
        endif()
    elseif(${LINALG_VENDOR} STREQUAL "IBMESSL")
        if(TARGET LAPACK_External)
            add_dependencies(GauXC_External LAPACK_External)
        endif()
    endif()
    if(USE_SCALAPACK)
        if(TARGET ScaLAPACK_External)
            add_dependencies(GauXC_External ScaLAPACK_External)
        endif()
    endif()
endif()
```

#### After Applying Fixes

Re-run cmake to regenerate build files:
```bash
cmake ..
make -j$(nproc)
make install
```

**Note:** These fixes are needed because the CMSB build system has a bug where it unconditionally adds dependencies on `*_External` targets even when those libraries are found pre-installed rather than built from source.

### Step 2c: Replace BuildTAMM.cmake to Use Local TAMM Source

Instead of cloning TAMM from GitHub, point the build to your local TAMM source directory. This way:
- All dependencies (LibInt2, GauXC, GlobalArrays, etc.) are built normally
- TAMM is built from your local source with your modifications
- Changes to local TAMM source are picked up on rebuild

**File location (after cmake configure):**
`build/_deps/cmakebuild-src/cmake/build_external/BuildTAMM.cmake`

**Replace entire file contents with:**
```cmake
#
# BuildTAMM.cmake - Modified to use local TAMM source directory
#
include(${CMAKE_CURRENT_LIST_DIR}/dep_versions.cmake)

# Path to local TAMM source (your modified version)
set(TAMM_LOCAL_SOURCE "/home/yash/Desktop/Applications/TAMM")

message(STATUS "Using local TAMM source from: ${TAMM_LOCAL_SOURCE}")

ExternalProject_Add(TAMM_External
    SOURCE_DIR ${TAMM_LOCAL_SOURCE}
    CMAKE_ARGS ${DEPENDENCY_CMAKE_OPTIONS}
    INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${STAGE_DIR}
    CMAKE_CACHE_ARGS ${CORE_CMAKE_LISTS} ${CORE_CMAKE_STRINGS}
    BUILD_ALWAYS TRUE
)
```

**Quick command to apply this fix:**
```bash
cat > build/_deps/cmakebuild-src/cmake/build_external/BuildTAMM.cmake << 'EOF'
#
# BuildTAMM.cmake - Modified to use local TAMM source directory
#
include(${CMAKE_CURRENT_LIST_DIR}/dep_versions.cmake)

# Path to local TAMM source (your modified version)
set(TAMM_LOCAL_SOURCE "/home/yash/Desktop/Applications/TAMM")

message(STATUS "Using local TAMM source from: ${TAMM_LOCAL_SOURCE}")

ExternalProject_Add(TAMM_External
    SOURCE_DIR ${TAMM_LOCAL_SOURCE}
    CMAKE_ARGS ${DEPENDENCY_CMAKE_OPTIONS}
    INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${STAGE_DIR}
    CMAKE_CACHE_ARGS ${CORE_CMAKE_LISTS} ${CORE_CMAKE_STRINGS}
    BUILD_ALWAYS TRUE
)
EOF
```

This modification:
- Uses your local TAMM source directory instead of cloning from GitHub
- All TAMM dependencies (LibInt2, GauXC, GlobalArrays, Librett, etc.) are built normally
- `BUILD_ALWAYS TRUE` ensures changes to TAMM source trigger rebuilds
- Your modifications to TAMM (e.g., gpu_blas.cpp) are automatically included

### Step 2d: Complete ExaChem Build Workflow (Summary)

Here's the complete workflow to build ExaChem with local TAMM source:

```bash
# 1. Clean and configure
cd /home/yash/Desktop/Applications/exachem
rm -rf build && mkdir build && cd build

CC=gcc CXX=g++ FC=gfortran cmake \
  -DCMAKE_INSTALL_PREFIX=$(pwd)/../install \
  -DTAMM_ENABLE_CUDA=ON \
  -DGPU_ARCH=61 \
  -DMODULES="CC" \
  ..

# 2. Apply BuildTAMM.cmake replacement (use local TAMM source)
cat > _deps/cmakebuild-src/cmake/build_external/BuildTAMM.cmake << 'EOF'
include(${CMAKE_CURRENT_LIST_DIR}/dep_versions.cmake)
set(TAMM_LOCAL_SOURCE "/home/yash/Desktop/Applications/TAMM")
message(STATUS "Using local TAMM source from: ${TAMM_LOCAL_SOURCE}")
ExternalProject_Add(TAMM_External
    SOURCE_DIR ${TAMM_LOCAL_SOURCE}
    CMAKE_ARGS ${DEPENDENCY_CMAKE_OPTIONS}
    INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${STAGE_DIR}
    CMAKE_CACHE_ARGS ${CORE_CMAKE_LISTS} ${CORE_CMAKE_STRINGS}
    BUILD_ALWAYS TRUE
)
EOF

# 3. Apply BuildGlobalArrays.cmake fix (if BLAS_External error occurs)
sed -i 's/add_dependencies(GlobalArrays_External BLAS_External LAPACK_External)/if(TARGET BLAS_External)\n            add_dependencies(GlobalArrays_External BLAS_External)\n        endif()\n        if(TARGET LAPACK_External)\n            add_dependencies(GlobalArrays_External LAPACK_External)\n        endif()/g' \
  _deps/cmakebuild-src/cmake/build_external/BuildGlobalArrays.cmake

# 4. Apply BuildGauXC.cmake fix (if needed)
sed -i 's/add_dependencies(GauXC_External BLAS_External LAPACK_External)/if(TARGET BLAS_External)\n        add_dependencies(GauXC_External BLAS_External)\n    endif()\n    if(TARGET LAPACK_External)\n        add_dependencies(GauXC_External LAPACK_External)\n    endif()/g' \
  _deps/cmakebuild-src/cmake/build_external/BuildGauXC.cmake

# 5. Re-run cmake and build
cmake ..
make -j$(nproc)
make install
```

### Step 3: After Modifying TAMM (Fast Rebuild)

```bash
# Rebuild TAMM only
cd TAMM/build
make -j$(nproc)
make install

# Relink ExaChem (fast - no full recompile)
cd ../../exachem/build
make -j$(nproc)
make install
```

### GPU_ARCH Values
| GPU | Architecture | GPU_ARCH |
|-----|--------------|----------|
| GTX 1080/1070 | Pascal | 61 |
| V100 | Volta | 70 |
| A100 | Ampere | 80 |
| H100 | Hopper | 90 |

---

## 3. Verify Installation

```bash
# Check executable
ls $EXACHEM_INSTALL/bin/ExaChem

# Test with a simple run
cd $EXACHEM_INSTALL
mpirun -np 2 ./bin/ExaChem --help
```

---

## 4. Input File Format

ExaChem uses JSON input files. Key sections:

```json
{
  "geometry": {
    "coordinates": ["atom x y z", ...],
    "units": "angstrom"
  },
  "basis": {
    "basisset": "cc-pvdz"
  },
  "SCF": { },
  "CD": { },
  "CC": { }
}
```

---

## 5. Python Runner Script

Save as `run_exachem.py`:

```python
#!/usr/bin/env python3
"""
ExaChem Python Runner
Generate input files and run ExaChem calculations.
Focuses on CCSD calculations which maximize TAMM backend usage.
"""

import json
import subprocess
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import tempfile
import shutil

# Configuration - adjust these paths
EXACHEM_BIN = Path.home() / "exachem_install/bin/ExaChem"
DEFAULT_SCRATCH = Path("/tmp/exachem_scratch")


class Molecule:
    """Molecular geometry container."""

    def __init__(self, name: str = "molecule"):
        self.name = name
        self.atoms: List[Tuple[str, float, float, float]] = []
        self.charge = 0
        self.multiplicity = 1

    def add_atom(self, symbol: str, x: float, y: float, z: float):
        """Add an atom to the molecule."""
        self.atoms.append((symbol, x, y, z))
        return self

    def from_xyz(self, xyz_string: str):
        """Parse XYZ format string."""
        lines = xyz_string.strip().split('\n')
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                symbol = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                self.add_atom(symbol, x, y, z)
        return self

    def get_coordinates(self) -> List[str]:
        """Get coordinates in ExaChem format."""
        return [f"{sym} {x:.6f} {y:.6f} {z:.6f}"
                for sym, x, y, z in self.atoms]

    @property
    def num_atoms(self) -> int:
        return len(self.atoms)


# Predefined test molecules
MOLECULES = {
    "h2": Molecule("H2").add_atom("H", 0.0, 0.0, 0.0).add_atom("H", 0.0, 0.0, 0.74),

    "water": Molecule("Water").add_atom("O", 0.0, 0.0, 0.117369)
                              .add_atom("H", 0.0, 0.757160, -0.469476)
                              .add_atom("H", 0.0, -0.757160, -0.469476),

    "methane": Molecule("Methane").add_atom("C", 0.0, 0.0, 0.0)
                                  .add_atom("H", 0.629118, 0.629118, 0.629118)
                                  .add_atom("H", -0.629118, -0.629118, 0.629118)
                                  .add_atom("H", -0.629118, 0.629118, -0.629118)
                                  .add_atom("H", 0.629118, -0.629118, -0.629118),

    "ammonia": Molecule("Ammonia").add_atom("N", 0.0, 0.0, 0.116489)
                                  .add_atom("H", 0.0, 0.939731, -0.271808)
                                  .add_atom("H", 0.813831, -0.469865, -0.271808)
                                  .add_atom("H", -0.813831, -0.469865, -0.271808),

    "benzene": Molecule("Benzene")
        .add_atom("C",  1.398, 0.000, 0.000)
        .add_atom("C",  0.699, 1.211, 0.000)
        .add_atom("C", -0.699, 1.211, 0.000)
        .add_atom("C", -1.398, 0.000, 0.000)
        .add_atom("C", -0.699, -1.211, 0.000)
        .add_atom("C",  0.699, -1.211, 0.000)
        .add_atom("H",  2.481, 0.000, 0.000)
        .add_atom("H",  1.240, 2.148, 0.000)
        .add_atom("H", -1.240, 2.148, 0.000)
        .add_atom("H", -2.481, 0.000, 0.000)
        .add_atom("H", -1.240, -2.148, 0.000)
        .add_atom("H",  1.240, -2.148, 0.000),

    "ethanol": Molecule("Ethanol")
        .add_atom("C", -1.271, -0.422, 0.000)
        .add_atom("C",  0.049,  0.312, 0.000)
        .add_atom("O",  1.131, -0.596, 0.000)
        .add_atom("H", -1.333, -1.064, 0.889)
        .add_atom("H", -1.333, -1.064, -0.889)
        .add_atom("H", -2.114,  0.281, 0.000)
        .add_atom("H",  0.097,  0.954, 0.889)
        .add_atom("H",  0.097,  0.954, -0.889)
        .add_atom("H",  1.972, -0.095, 0.000),
}


def create_input(
    molecule: Molecule,
    basis: str = "cc-pvdz",
    method: str = "ccsd",
    nprocs: int = 1,
    memory_gb: float = 2.0,
    scratch_dir: Optional[Path] = None,
    extra_options: Optional[dict] = None
) -> dict:
    """
    Create ExaChem input dictionary.

    Args:
        molecule: Molecule object with geometry
        basis: Basis set (cc-pvdz, cc-pvtz, aug-cc-pvdz, etc.)
        method: Calculation method (scf, mp2, ccsd, ccsd_t)
        nprocs: Number of MPI processes (for memory estimation)
        memory_gb: Memory per process in GB
        scratch_dir: Scratch directory for temporary files
        extra_options: Additional options to merge

    Returns:
        Input dictionary for ExaChem
    """

    input_dict = {
        "geometry": {
            "coordinates": molecule.get_coordinates(),
            "units": "angstrom"
        },
        "basis": {
            "basisset": basis
        },
        "common": {
            "maxiter": 100
        },
        "SCF": {
            "tol_int": 1e-12,
            "tol_lindep": 1e-6,
            "conve": 1e-9,
            "convd": 1e-8,
            "diis_hist": 10
        },
        "CD": {
            "diagtol": 1e-6
        }
    }

    # Add method-specific options
    if method.lower() in ["ccsd", "ccsd_t", "ccsd(t)"]:
        input_dict["CC"] = {
            "threshold": 1e-6,
            "maxiter": 100,
            "CCSD(T)": {
                "compute": method.lower() in ["ccsd_t", "ccsd(t)"]
            }
        }
    elif method.lower() == "mp2":
        input_dict["CD"]["compute_mp2"] = True

    # Add scratch directory
    if scratch_dir:
        input_dict["common"]["scratch_dir"] = str(scratch_dir)

    # Merge extra options
    if extra_options:
        for key, value in extra_options.items():
            if key in input_dict and isinstance(value, dict):
                input_dict[key].update(value)
            else:
                input_dict[key] = value

    return input_dict


def run_exachem(
    input_dict: dict,
    output_dir: Path,
    nprocs: int = 1,
    exachem_bin: Path = EXACHEM_BIN,
    timeout: Optional[int] = None,
    verbose: bool = True,
    profile: bool = False,
    gemm_dtype: str = "fp64",
    gemm_csv: Optional[str] = None
) -> subprocess.CompletedProcess:
    """
    Run ExaChem calculation.

    Args:
        input_dict: Input dictionary
        output_dir: Directory for output files
        nprocs: Number of MPI processes
        exachem_bin: Path to ExaChem executable
        timeout: Timeout in seconds
        verbose: Print output in real-time
        profile: Enable TAMM GEMM profiling
        gemm_dtype: GEMM compute precision (fp64, fp32, fp16, bf16, tf32)
        gemm_csv: Output CSV path for profiling (default: output_dir/gemm_profile.csv)

    Returns:
        CompletedProcess object
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write input file
    input_file = output_dir / "input.json"
    with open(input_file, 'w') as f:
        json.dump(input_dict, f, indent=2)

    if verbose:
        print(f"Input file: {input_file}")
        print(f"Running with {nprocs} MPI processes...")

    # Set up environment with profiling variables
    env = os.environ.copy()
    if profile:
        env["TAMM_GEMM_PROFILE"] = "1"
        env["TAMM_GEMM_DTYPE"] = gemm_dtype
        csv_path = gemm_csv if gemm_csv else str(output_dir / "gemm_profile.csv")
        env["TAMM_GEMM_CSV"] = csv_path
        if verbose:
            print(f"Profiling enabled: dtype={gemm_dtype}, csv={csv_path}")

    # Build command
    cmd = ["mpirun", "-np", str(nprocs), str(exachem_bin), str(input_file)]

    # Run
    if verbose:
        # Real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(output_dir),
            env=env,
            text=True
        )

        output_lines = []
        for line in process.stdout:
            print(line, end='')
            output_lines.append(line)

        process.wait()

        result = subprocess.CompletedProcess(
            cmd, process.returncode,
            stdout=''.join(output_lines),
            stderr=''
        )
    else:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(output_dir),
            env=env
        )

    # Save output
    output_file = output_dir / "output.log"
    with open(output_file, 'w') as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)

    return result


def run_molecule(
    name: str,
    basis: str = "cc-pvdz",
    method: str = "ccsd",
    nprocs: int = 2,
    output_base: Path = Path("./exachem_runs"),
    profile: bool = False,
    gemm_dtype: str = "fp64",
    gemm_csv: Optional[str] = None,
    **kwargs
) -> subprocess.CompletedProcess:
    """
    Convenience function to run a predefined molecule.

    Args:
        name: Molecule name (h2, water, methane, ammonia, benzene, ethanol)
        basis: Basis set
        method: Calculation method
        nprocs: Number of MPI processes
        output_base: Base directory for outputs
        profile: Enable TAMM GEMM profiling
        gemm_dtype: GEMM compute precision (fp64, fp32, fp16, bf16, tf32)
        gemm_csv: Output CSV path for profiling
        **kwargs: Additional options for create_input

    Returns:
        CompletedProcess object
    """

    if name not in MOLECULES:
        raise ValueError(f"Unknown molecule: {name}. Available: {list(MOLECULES.keys())}")

    molecule = MOLECULES[name]
    output_dir = output_base / f"{name}_{basis}_{method}"

    input_dict = create_input(
        molecule=molecule,
        basis=basis,
        method=method,
        nprocs=nprocs,
        **kwargs
    )

    print(f"\n{'='*60}")
    print(f"Running: {molecule.name}")
    print(f"Atoms: {molecule.num_atoms}, Basis: {basis}, Method: {method.upper()}")
    if profile:
        print(f"Profiling: dtype={gemm_dtype}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    return run_exachem(input_dict, output_dir, nprocs,
                       profile=profile, gemm_dtype=gemm_dtype, gemm_csv=gemm_csv)


def run_custom_xyz(
    xyz_string: str,
    name: str = "custom",
    basis: str = "cc-pvdz",
    method: str = "ccsd",
    nprocs: int = 2,
    output_base: Path = Path("./exachem_runs"),
    profile: bool = False,
    gemm_dtype: str = "fp64",
    gemm_csv: Optional[str] = None,
    **kwargs
) -> subprocess.CompletedProcess:
    """
    Run calculation with custom XYZ geometry.

    Args:
        xyz_string: XYZ format geometry string
        name: Molecule name for output directory
        basis: Basis set
        method: Calculation method
        nprocs: Number of MPI processes
        output_base: Base directory for outputs
        profile: Enable TAMM GEMM profiling
        gemm_dtype: GEMM compute precision (fp64, fp32, fp16, bf16, tf32)
        gemm_csv: Output CSV path for profiling

    Returns:
        CompletedProcess object
    """

    molecule = Molecule(name).from_xyz(xyz_string)
    output_dir = output_base / f"{name}_{basis}_{method}"

    input_dict = create_input(
        molecule=molecule,
        basis=basis,
        method=method,
        nprocs=nprocs,
        **kwargs
    )

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Atoms: {molecule.num_atoms}, Basis: {basis}, Method: {method.upper()}")
    if profile:
        print(f"Profiling: dtype={gemm_dtype}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    return run_exachem(input_dict, output_dir, nprocs,
                       profile=profile, gemm_dtype=gemm_dtype, gemm_csv=gemm_csv)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ExaChem calculations")
    parser.add_argument("molecule", nargs="?", default="water",
                        help=f"Molecule name: {list(MOLECULES.keys())}")
    parser.add_argument("-b", "--basis", default="cc-pvdz",
                        help="Basis set (default: cc-pvdz)")
    parser.add_argument("-m", "--method", default="ccsd",
                        choices=["scf", "mp2", "ccsd", "ccsd_t"],
                        help="Calculation method (default: ccsd)")
    parser.add_argument("-n", "--nprocs", type=int, default=2,
                        help="Number of MPI processes (default: 2)")
    parser.add_argument("-o", "--output", default="./exachem_runs",
                        help="Output directory (default: ./exachem_runs)")
    parser.add_argument("--list", action="store_true",
                        help="List available molecules")
    parser.add_argument("--xyz", type=str,
                        help="Path to XYZ file for custom molecule")

    # Profiling options
    parser.add_argument("--profile", action="store_true",
                        help="Enable TAMM GEMM profiling")
    parser.add_argument("--dtype", default="fp64",
                        choices=["fp64", "fp32", "fp16", "bf16", "tf32"],
                        help="GEMM compute precision (default: fp64)")
    parser.add_argument("--csv", type=str,
                        help="Output CSV path for profiling (default: output_dir/gemm_profile.csv)")

    args = parser.parse_args()

    if args.list:
        print("Available molecules:")
        for name, mol in MOLECULES.items():
            print(f"  {name:12} - {mol.num_atoms} atoms")
        sys.exit(0)

    # Check ExaChem binary
    if not EXACHEM_BIN.exists():
        print(f"Error: ExaChem not found at {EXACHEM_BIN}")
        print("Update EXACHEM_BIN in this script or build ExaChem first.")
        sys.exit(1)

    output_base = Path(args.output)

    if args.xyz:
        # Run with custom XYZ file
        with open(args.xyz) as f:
            xyz_content = f.read()
        name = Path(args.xyz).stem
        result = run_custom_xyz(
            xyz_content, name=name,
            basis=args.basis, method=args.method,
            nprocs=args.nprocs, output_base=output_base,
            profile=args.profile, gemm_dtype=args.dtype, gemm_csv=args.csv
        )
    else:
        # Run predefined molecule
        result = run_molecule(
            args.molecule,
            basis=args.basis, method=args.method,
            nprocs=args.nprocs, output_base=output_base,
            profile=args.profile, gemm_dtype=args.dtype, gemm_csv=args.csv
        )

    print(f"\nExit code: {result.returncode}")

    # Print profiling summary if enabled
    if args.profile:
        csv_path = args.csv if args.csv else output_base / f"{args.molecule if not args.xyz else name}_{args.basis}_{args.method}" / "gemm_profile.csv"
        if Path(csv_path).exists():
            print(f"\nProfiler CSV: {csv_path}")
            print("Analyze with: python -c \"import pandas as pd; df=pd.read_csv('{csv_path}'); print(df.describe())\"")
```

---

## 6. Usage Examples

### Run from command line:

```bash
# Simple water calculation
python run_exachem.py water

# Benzene with larger basis set (more TAMM operations)
python run_exachem.py benzene -b cc-pvtz -m ccsd -n 4

# CCSD(T) calculation (heaviest TAMM usage)
python run_exachem.py methane -m ccsd_t -n 4

# Custom molecule from XYZ file
python run_exachem.py --xyz my_molecule.xyz -b aug-cc-pvdz -m ccsd

# With GEMM profiling enabled (default fp64)
python run_exachem.py water --profile

# Profile with TF32 precision (faster on Ampere GPUs)
python run_exachem.py benzene -b cc-pvdz -m ccsd --profile --dtype tf32

# Compare different precisions
for dtype in fp64 fp32 tf32; do
    python run_exachem.py water --profile --dtype $dtype --csv results_${dtype}.csv
done
```

### Run from Python:

```python
from run_exachem import run_molecule, run_custom_xyz, Molecule, create_input, run_exachem

# Quick test
run_molecule("water", basis="cc-pvdz", method="ccsd", nprocs=2)

# Larger calculation (more TAMM time)
run_molecule("benzene", basis="cc-pvtz", method="ccsd", nprocs=4)

# With profiling enabled
run_molecule("water", basis="cc-pvdz", method="ccsd", nprocs=2,
             profile=True, gemm_dtype="tf32")

# Custom molecule
xyz = """
C  0.000  0.000  0.000
O  1.200  0.000  0.000
"""
run_custom_xyz(xyz, name="carbon_monoxide", basis="cc-pvdz", method="ccsd",
               profile=True, gemm_dtype="fp64", gemm_csv="./co_profile.csv")

# Full control with profiling
mol = Molecule("my_mol")
mol.add_atom("N", 0, 0, 0)
mol.add_atom("N", 0, 0, 1.1)

input_dict = create_input(mol, basis="cc-pvtz", method="ccsd")
run_exachem(input_dict, output_dir="./n2_run", nprocs=4,
            profile=True, gemm_dtype="tf32")
```

---

## 7. Maximizing TAMM Backend Usage

To spend **maximum time in TAMM tensor operations**:

| Factor | More TAMM Time |
|--------|----------------|
| **Method** | CCSD(T) > CCSD > MP2 > SCF |
| **Basis Set** | cc-pvtz > cc-pvdz > cc-pvdz |
| **Molecule Size** | Larger = more tensor operations |
| **Electrons** | More correlated electrons = bigger tensors |

### Recommended test cases:

```bash
# Light (seconds) - testing
python run_exachem.py water -b cc-pvdz -m ccsd

# Medium (minutes) - benchmarking
python run_exachem.py benzene -b cc-pvdz -m ccsd

# Heavy (hours) - stress testing
python run_exachem.py benzene -b cc-pvtz -m ccsd_t
```

---

## 8. Profiling TAMM Operations

ExaChem/TAMM outputs timing information. Look for in output:

```
CCSD iteration times
Total CCSD time
Tensor contraction time
```

For detailed TAMM profiling, rebuild with:

```bash
cmake ... -DUSE_GA_PROFILER=ON
```

---

## 9. Troubleshooting

| Error | Solution |
|-------|----------|
| `MPI not found` | `sudo apt install openmpi-bin libopenmpi-dev` |
| `BLAS not found` | `sudo apt install libblas-dev liblapack-dev` |
| `Out of memory` | Reduce basis set or molecule size |
| `Convergence failed` | Increase `maxiter` or adjust `diis_hist` |

---

## 10. TAMM CUDA GEMM Profiler & Precision Switching

Custom modifications to TAMM's `gpu_blas.cpp` enable:
- **Profiling**: CSV logging of all GEMM calls with timing
- **Precision switching**: Change compute precision via environment variable

### 10.1 Modified File Location

```
TAMM/src/tamm/kernels/gpu_blas.cpp
```

### 10.2 Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `TAMM_GEMM_PROFILE` | `0` or `1` | Enable/disable GEMM profiling (default: 0) |
| `TAMM_GEMM_DTYPE` | `fp64`, `fp32`, `fp16`, `bf16`, `tf32` | GEMM compute precision (default: fp64) |
| `TAMM_GEMM_CSV` | filepath | Output CSV path (default: `tamm_gemm_profile.csv`) |

### 10.3 Precision Types

| Type | Description | GPU Requirement |
|------|-------------|-----------------|
| `fp64` | Double precision (64-bit) | Any CUDA GPU |
| `fp32` | Single precision (32-bit) | Any CUDA GPU |
| `fp16` | Half precision (16-bit) | Pascal+ (SM 60+) |
| `bf16` | BFloat16 (16-bit) | Ampere+ (SM 80+) |
| `tf32` | TensorFloat-32 | Ampere+ (SM 80+) |

### 10.4 CSV Output Format

```csv
m,n,k,dtype,time_us,gflops
128,256,512,fp64,1234.56,78.9
256,256,256,tf32,456.78,234.5
```

### 10.5 Usage Examples

```bash
# Enable profiling with default FP64
export TAMM_GEMM_PROFILE=1
mpirun -np 4 ExaChem input.json

# Use TF32 precision (faster on Ampere GPUs)
export TAMM_GEMM_PROFILE=1
export TAMM_GEMM_DTYPE=tf32
export TAMM_GEMM_CSV=./results/gemm_tf32.csv
mpirun -np 4 ExaChem input.json

# Compare precisions
for dtype in fp64 fp32 tf32 fp16; do
  export TAMM_GEMM_DTYPE=$dtype
  export TAMM_GEMM_CSV="gemm_${dtype}.csv"
  mpirun -np 4 ExaChem input.json
done
```

---

## 11. Rebuilding After TAMM Modifications

After modifying TAMM code (e.g., `gpu_blas.cpp`), follow these steps based on your build approach.

### 11.1 External TAMM Build (Recommended)

If you followed Section 2.2:

```bash
# 1. Rebuild TAMM
cd TAMM/build
make -j$(nproc)
make install

# 2. Relink ExaChem (fast)
cd ../../exachem/build
make -j$(nproc)
make install
```

### 11.2 Embedded TAMM Build

If you followed Section 2.1 (TAMM is inside ExaChem):

```bash
# Navigate to TAMM within ExaChem
cd exachem/contrib/TAMM

# Make your modifications
vim src/tamm/kernels/gpu_blas.cpp

# Rebuild ExaChem (slower - rebuilds everything)
cd ../../build
make -j$(nproc)
make install
```

### 11.3 Full Clean Rebuild (if CMake issues)

```bash
cd TAMM
rm -rf build && mkdir build && cd build

CC=gcc CXX=g++ FC=gfortran cmake \
  -DCMAKE_INSTALL_PREFIX=$(pwd)/../install \
  -DTAMM_ENABLE_CUDA=ON \
  -DGPU_ARCH=61 \
  -DALLOW_CONDA=ON \
  ..

make -j$(nproc)
make install

# Then rebuild ExaChem
cd ../../exachem/build
rm -rf *

TAMM_INSTALL=/home/yash/Desktop/Applications/TAMM/install

CC=gcc CXX=g++ FC=gfortran cmake \
  -DCMAKE_INSTALL_PREFIX=$(pwd)/../install \
  -DCMAKE_PREFIX_PATH=$TAMM_INSTALL \
  -DCMSB_DEBUG_CMAKE=ON \
  -DTAMM_ENABLE_CUDA=ON \
  -DGPU_ARCH=61 \
  -DALLOW_CONDA=ON \
  -DMODULES="CC" \
  ..

make -j$(nproc)
make install
```

---

## 12. Why External TAMM is Preferred for Development

| Aspect | Embedded TAMM | External TAMM |
|--------|---------------|---------------|
| **Initial Setup** | Simpler (one clone) | Two separate clones |
| **TAMM Modifications** | Edit in `exachem/contrib/TAMM/` | Edit in `TAMM/` |
| **Rebuild After Changes** | Full ExaChem rebuild | TAMM rebuild + relink |
| **Rebuild Time** | 30-60 minutes | 5-10 minutes |
| **Version Control** | Submodule complications | Clean separate repos |

**Note**: You cannot swap TAMM in a prebuilt ExaChem binary. TAMM is statically linked at compile time, so any TAMM modifications require rebuilding.

---

## 13. Analyzing Profiler Output

### 13.1 Python Analysis Script

```python
#!/usr/bin/env python3
"""Analyze TAMM GEMM profiler CSV output."""

import pandas as pd
import matplotlib.pyplot as plt

def analyze_gemm_profile(csv_path: str):
    df = pd.read_csv(csv_path)

    print(f"Total GEMM calls: {len(df)}")
    print(f"Total time: {df['time_us'].sum() / 1e6:.2f} seconds")
    print(f"Average GFLOPS: {df['gflops'].mean():.2f}")
    print(f"\nTime distribution:")
    print(df['time_us'].describe())

    # Top 10 slowest GEMMs
    print("\nTop 10 slowest GEMM calls:")
    print(df.nlargest(10, 'time_us')[['m', 'n', 'k', 'time_us', 'gflops']])

    # Plot histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df['time_us'], bins=50, edgecolor='black')
    axes[0].set_xlabel('Time (microseconds)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('GEMM Latency Distribution')

    axes[1].scatter(df['m'] * df['n'] * df['k'], df['gflops'], alpha=0.5)
    axes[1].set_xlabel('Problem Size (M*N*K)')
    axes[1].set_ylabel('GFLOPS')
    axes[1].set_title('GFLOPS vs Problem Size')

    plt.tight_layout()
    plt.savefig('gemm_analysis.png', dpi=150)
    print("\nSaved plot to gemm_analysis.png")

if __name__ == "__main__":
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "tamm_gemm_profile.csv"
    analyze_gemm_profile(csv_file)
```

### 13.2 Quick Analysis with Shell

```bash
# Count total GEMM calls
wc -l tamm_gemm_profile.csv

# Sum total time (microseconds)
awk -F',' 'NR>1 {sum+=$5} END {print "Total time (sec):", sum/1e6}' tamm_gemm_profile.csv

# Find largest GEMM calls
sort -t',' -k5 -rn tamm_gemm_profile.csv | head -10

# Average GFLOPS
awk -F',' 'NR>1 {sum+=$6; n++} END {print "Avg GFLOPS:", sum/n}' tamm_gemm_profile.csv
```

---

## 14. Code Changes Reference

### 14.1 Modified Files

| File | Changes |
|------|---------|
| `TAMM/src/tamm/kernels/gpu_blas.cpp` | Added GemmProfiler class, dtype switching, timing |

### 14.2 Key Code Sections

**GemmProfiler Class** (singleton):
- Reads `TAMM_GEMM_PROFILE`, `TAMM_GEMM_DTYPE`, `TAMM_GEMM_CSV` env vars
- Thread-safe CSV logging with mutex
- Calculates GFLOPS from m,n,k,time

**GEMM Function Changes**:
- Added `cudaDeviceSynchronize()` before/after for accurate timing
- Switch statement for dtype selection
- Uses `cublasGemmEx` for mixed precision modes
- Uses `cublasSetMathMode` for TF32/BF16 tensor ops

### 14.3 cuBLAS Functions Used

| Precision | cuBLAS Function | Compute Type |
|-----------|-----------------|--------------|
| FP64 | `cublasDgemm` | CUDA_R_64F |
| FP32 | `cublasGemmEx` | CUBLAS_COMPUTE_32F |
| FP16 | `cublasGemmEx` | CUBLAS_COMPUTE_16F |
| TF32 | `cublasGemmEx` | CUBLAS_COMPUTE_32F_FAST_TF32 |
| BF16 | `cublasGemmEx` | CUBLAS_COMPUTE_32F + TENSOR_OP |

---

## 15. Performance Notes

### 15.1 Expected Speedups (vs FP64)

| Precision | Typical Speedup | Accuracy Loss |
|-----------|-----------------|---------------|
| TF32 | 2-3x | ~0.1% |
| FP32 | 2x | Variable |
| FP16 | 4-8x | Significant |
| BF16 | 2-4x | Moderate |

### 15.2 Recommendations

- **Production**: Use `fp64` (default) for scientific accuracy
- **Benchmarking**: Use `tf32` on Ampere+ for speed with minimal accuracy loss
- **Profiling overhead**: `cudaDeviceSynchronize` adds latency; disable profiling for production runs

### 15.3 GPU Architecture Requirements

```
Pascal (GTX 10xx, SM 60-61): fp64, fp32, fp16
Volta (V100, SM 70): + Tensor Cores for fp16
Turing (RTX 20xx, SM 75): + INT8
Ampere (A100, RTX 30xx, SM 80+): + TF32, BF16
Hopper (H100, SM 90+): + FP8
```
