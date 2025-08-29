import psutil
from dask.distributed import Client
import dask

def setup_optimal_client(
    backend: str = "numba_cpu",
    memory_fraction: float = 0.8,
    cpu_fraction: float = 0.9,
    verbose: bool = True
) -> Client:
    """Setup optimal Dask client based on computer specifications.
    
    Automatically configures workers, threads, and memory limits based on
    the computer's hardware specifications using psutil.
    
    Parameters
    ----------
    backend : str, optional
        Backend type: "numpy", "numba_cpu", "cupy", or "auto"
    memory_fraction : float, optional
        Fraction of total RAM to use (default: 0.8 = 80%)
    cpu_fraction : float, optional
        Fraction of CPU cores to use (default: 0.9 = 90%)
    verbose : bool, optional
        Whether to print configuration details
    
    Returns
    -------
    Client
        Configured Dask client
    """

    # Get system specifications
    cpu_count = psutil.cpu_count(logical=True)  # Logical cores
    cpu_count_physical = psutil.cpu_count(logical=False)  # Physical cores
    memory_total = psutil.virtual_memory().total
    memory_available = psutil.virtual_memory().available

    # Convert memory to GB
    memory_total_gb = memory_total / (1024**3)
    memory_available_gb = memory_available / (1024**3)

    if verbose:
        print(f"System Specifications:")
        print(f"  Physical Cores: {cpu_count_physical}")
        print(f"  Logical Cores: {cpu_count}")
        print(f"  Total RAM: {memory_total_gb:.1f} GB")
        print(f"  Available RAM: {memory_available_gb:.1f} GB")

    # Calculate optimal configuration based on backend
    if backend == "auto":
        # Auto-detect based on system specs
        if memory_total_gb >= 32 and cpu_count >= 8:
            backend = "numba_cpu"
        elif memory_total_gb >= 16:
            backend = "numpy"
        else:
            backend = "numba_cpu"  # Default to memory-efficient

    # Configure based on backend
    if backend == "numpy":
        # NumPy: Use processes to avoid memory sharing issues
        # Conservative approach for memory-intensive operations
        n_workers = max(1, min(4, round(cpu_count_physical * cpu_fraction)))
        threads_per_worker = 1  # NumPy is single-threaded
        processes = True
        memory_limit_per_worker = f"{round(memory_available_gb * memory_fraction / n_workers)}GB"

    elif backend == "numba_cpu":
        # Numba: Can use threads for better memory efficiency
        # Use physical cores for processes, logical cores for threads
        n_workers = max(1, round(cpu_count_physical * cpu_fraction))
        threads_per_worker = max(1, round(cpu_count / cpu_count_physical * cpu_fraction))
        processes = True  # Use processes for better isolation
        memory_limit_per_worker = f"{round(memory_available_gb * memory_fraction / n_workers)}GB"

    elif backend == "cupy":
        # CuPy: Use processes for GPU isolation
        # Conservative approach for GPU operations
        n_workers = max(1, min(2, round(cpu_count_physical * cpu_fraction)))
        threads_per_worker = 1  # Single thread per GPU worker
        processes = True
        memory_limit_per_worker = f"{round(memory_available_gb * memory_fraction / n_workers)}GB"

    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Create client configuration
    client_config = {
        "processes": processes,
        "n_workers": n_workers,
        "threads_per_worker": threads_per_worker,
        "memory_limit": memory_limit_per_worker,
        "silence_logs": 50,  # Reduce logging noise
    }

    # Additional optimizations using current Dask config
    if backend == "numba_cpu":
        # Set Dask configuration for memory management
        dask.config.set(
            {
                "distributed.worker.memory.target": 0.6,  # Target memory usage (60%)
                "distributed.worker.memory.spill": 0.8,  # Spill to disk at 80%
                "distributed.worker.memory.pause": 0.95,  # Pause at 95%
                "distributed.worker.memory.terminate": 0.98,  # Terminate at 98%
            }
        )

        # Add local directory for spill files
        client_config.update(
            {
                "local_directory": "/tmp/dask-worker-space",  # Local temp directory
            }
        )

    if verbose:
        print(f"\nDask Configuration for {backend} backend:")
        print(f"  Processes: {processes}")
        print(f"  Workers: {n_workers}")
        print(f"  Threads per worker: {threads_per_worker}")
        print(f"  Memory limit per worker: {memory_limit_per_worker}")
        print(f"  Total threads: {n_workers * threads_per_worker}")
        print(
            f"  Total memory allocation: {n_workers * round(memory_available_gb * memory_fraction / n_workers)} GB"
        )

        if backend == "numba_cpu":
            print(f"  Memory target: 60%")
            print(f"  Memory spill threshold: 80%")
            print(f"  Memory pause threshold: 95%")

    # Create and return client
    client = Client(**client_config)

    if verbose:
        print(f"\nClient created successfully!")
        print(f"  Dashboard: {client.dashboard_link}")
        print(f"  Workers: {len(client.scheduler_info()['workers'])}")

    return client