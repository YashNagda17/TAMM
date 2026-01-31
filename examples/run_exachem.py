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
EXACHEM_BIN = Path("/home/yash/Desktop/Applications/install/bin/ExaChem")
DEFAULT_SCRATCH = Path("/tmp/exachem_scratch")
DEFAULT_CSV_DIR = Path.cwd()  # CSV files stored in current working directory


def get_gpu_count() -> int:
    """Detect number of available NVIDIA GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return len(result.stdout.strip().split('\n'))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0


def setup_gpu_env(
    gpu_ids: Optional[List[int]] = None,
    csv_file: Optional[Path] = None
) -> dict:
    """
    Set up environment variables for GPU execution with TAMM profiling.

    Args:
        gpu_ids: Specific GPU IDs to use, or None for all available
        csv_file: Path to CSV file for TAMM GEMM profiling output

    Returns:
        Environment dict with GPU settings
    """
    env = os.environ.copy()

    num_gpus = get_gpu_count()
    if num_gpus == 0:
        print("WARNING: No GPUs detected! Falling back to CPU execution.")
        return env

    if gpu_ids is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        print(f"Using GPUs: {gpu_ids}")
    else:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(num_gpus)))
        print(f"Using all {num_gpus} GPU(s)")

    # Optimize GPU memory allocation
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # TAMM/GA settings for GPU
    env["GA_NUM_PROGRESS_RANKS"] = "0"

    # CUBLAS settings for optimal GPU performance
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["CUDA_LAUNCH_BLOCKING"] = "0"

    # TAMM GEMM profiling settings
    env["TAMM_GEMM_PROFILE"] = "1"
    if csv_file is not None:
        env["TAMM_GEMM_CSV"] = str(csv_file)
        print(f"TAMM GEMM CSV: {csv_file}")

    return env


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

    # Add method-specific options and TASK (only one task can be enabled)
    if method.lower() == "ccsd":
        input_dict["CC"] = {
            "threshold": 1e-6
        }
        input_dict["TASK"] = {"ccsd": True}
    elif method.lower() in ["ccsd_t", "ccsd(t)"]:
        input_dict["CC"] = {
            "threshold": 1e-6,
            "CCSD(T)": {
                "ccsdt_tilesize": 28
            }
        }
        input_dict["TASK"] = {"ccsd_t": True}
    elif method.lower() == "mp2":
        input_dict["TASK"] = {"mp2": True}
    elif method.lower() == "scf":
        input_dict["TASK"] = {"scf": True}

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
    gpu_ids: Optional[List[int]] = None,
    molecule_name: str = "molecule",
    basis: str = "cc-pvdz",
    csv_dir: Optional[Path] = None
) -> subprocess.CompletedProcess:
    """
    Run ExaChem calculation on GPU.

    Args:
        input_dict: Input dictionary
        output_dir: Directory for output files
        nprocs: Number of MPI processes (should match GPU count for optimal performance)
        exachem_bin: Path to ExaChem executable
        timeout: Timeout in seconds
        verbose: Print output in real-time
        gpu_ids: Specific GPU IDs to use, or None for all available
        molecule_name: Name of the molecule (for CSV filename)
        basis: Basis set used (for CSV filename)
        csv_dir: Directory for CSV output (default: current working directory)

    Returns:
        CompletedProcess object
    """

    output_dir = Path(output_dir).resolve()  # Use absolute path
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up CSV file path for TAMM GEMM profiling
    csv_dir = Path(csv_dir).resolve() if csv_dir else DEFAULT_CSV_DIR.resolve()
    csv_file = csv_dir / f"{molecule_name}_{basis}.csv"

    # Set up GPU environment with TAMM profiling
    # Note: MPI processes run on CPU, TAMM GEMM calls offload to GPU
    env = setup_gpu_env(gpu_ids, csv_file=csv_file)

    # Ensure minimum 2 MPI ranks (required by ExaChem)
    if nprocs < 2:
        print(f"Adjusting MPI processes from {nprocs} to 2 (minimum required)")
        nprocs = 2

    # Write input file
    input_file = output_dir / "input.json"
    with open(input_file, 'w') as f:
        json.dump(input_dict, f, indent=2)

    if verbose:
        print(f"Input file: {input_file}")
        print(f"Running with {nprocs} MPI process(es) on GPU...")

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
            text=True,
            env=env
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
    nprocs: int = 1,
    output_base: Path = Path("./exachem_runs"),
    gpu_ids: Optional[List[int]] = None,
    **kwargs
) -> subprocess.CompletedProcess:
    """
    Convenience function to run a predefined molecule on GPU.

    Args:
        name: Molecule name (h2, water, methane, ammonia, benzene, ethanol)
        basis: Basis set
        method: Calculation method
        nprocs: Number of MPI processes (auto-adjusted to GPU count)
        output_base: Base directory for outputs
        gpu_ids: Specific GPU IDs to use, or None for all available
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
    print(f"Running: {molecule.name} [GPU]")
    print(f"Atoms: {molecule.num_atoms}, Basis: {basis}, Method: {method.upper()}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    return run_exachem(
        input_dict, output_dir, nprocs,
        gpu_ids=gpu_ids,
        molecule_name=name,
        basis=basis
    )


def run_custom_xyz(
    xyz_string: str,
    name: str = "custom",
    basis: str = "cc-pvdz",
    method: str = "ccsd",
    nprocs: int = 1,
    output_base: Path = Path("./exachem_runs"),
    gpu_ids: Optional[List[int]] = None,
    **kwargs
) -> subprocess.CompletedProcess:
    """
    Run calculation with custom XYZ geometry on GPU.

    Args:
        xyz_string: XYZ format geometry string
        name: Molecule name for output directory
        basis: Basis set
        method: Calculation method
        nprocs: Number of MPI processes (auto-adjusted to GPU count)
        output_base: Base directory for outputs
        gpu_ids: Specific GPU IDs to use, or None for all available

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
    print(f"Running: {name} [GPU]")
    print(f"Atoms: {molecule.num_atoms}, Basis: {basis}, Method: {method.upper()}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    return run_exachem(
        input_dict, output_dir, nprocs,
        gpu_ids=gpu_ids,
        molecule_name=name,
        basis=basis
    )


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
            nprocs=args.nprocs, output_base=output_base
        )
    else:
        # Run predefined molecule
        result = run_molecule(
            args.molecule,
            basis=args.basis, method=args.method,
            nprocs=args.nprocs, output_base=output_base
        )

    print(f"\nExit code: {result.returncode}")
