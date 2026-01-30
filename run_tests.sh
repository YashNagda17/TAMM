#!/bin/bash
# TAMM Test Runner with GEMM Profiling
# Builds TAMM and runs tests, recording GEMM latencies to CSV

set -e

# Get directory where this script is located
TAMM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$TAMM_DIR/build"
CSV_DIR="$TAMM_DIR/csv"
GPU_ARCH=${GPU_ARCH:-61}

# Create CSV output directory
mkdir -p "$CSV_DIR"

# Disable conda
conda deactivate 2>/dev/null || true
unset CONDA_EXE CONDA_PREFIX CONDA_PYTHON_EXE CONDA_DEFAULT_ENV CONDA_SHLVL

# CUDA setup
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "=========================================="
echo "TAMM Test Runner"
echo "=========================================="
echo "TAMM_DIR: $TAMM_DIR"
echo "GPU_ARCH: $GPU_ARCH"
echo "CSV_DIR:  $CSV_DIR"
echo ""

# Step 1: Build TAMM
echo "[1/3] Building TAMM..."
cd "$TAMM_DIR"
rm -rf build && mkdir build && cd build

CC=gcc CXX=g++ FC=gfortran cmake \
  -DCMAKE_INSTALL_PREFIX="$TAMM_DIR/install" \
  -DTAMM_ENABLE_CUDA=ON \
  -DGPU_ARCH=$GPU_ARCH \
  -DALLOW_CONDA=ON \
  ..

make -j$(nproc-2) && make install
echo "Build complete."
echo ""

# Step 2: Run tests with profiling
echo "[2/3] Running TAMM tests with GEMM profiling..."

# Enable GEMM profiling
export TAMM_GEMM_PROFILE=1

# Test configurations: (N, tile_size)
TESTS=(
  "50 20"
  "100 25"
  "200 50"
)

cd "$BUILD_DIR/tests/tamm"

for test_args in "${TESTS[@]}"; do
  read -r N TILE <<< "$test_args"
  CSV_FILE="$CSV_DIR/gemm_profile_${N}_${TILE}.csv"
  export TAMM_GEMM_CSV="$CSV_FILE"

  echo "  Running Test_Mult_Ops N=$N tile=$TILE -> $CSV_FILE"
  mpirun -np 2 ./Test_Mult_Ops $N $TILE 2>&1 | tee "$CSV_DIR/test_output_${N}_${TILE}.log" || true
done

# Step 3: Summary
echo ""
echo "[3/3] Test Summary"
echo "=========================================="
echo "CSV files generated in: $CSV_DIR"
ls -la "$CSV_DIR"/*.csv 2>/dev/null || echo "No CSV files found"

echo ""
echo "Sample CSV content:"
for csv in "$CSV_DIR"/*.csv; do
  if [ -f "$csv" ]; then
    echo ""
    echo "--- $(basename $csv) ---"
    head -10 "$csv"
  fi
done

echo ""
echo "Done."