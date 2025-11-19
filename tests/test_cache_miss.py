#!/usr/bin/env python3
"""
Test script to verify cache miss monitoring is working correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import time
import numpy as np
from benchmark.sensors.cache_miss import get_cache_monitor

def simulate_anns_operation():
    """Simulate memory access patterns similar to ANNS algorithms"""
    # Create dataset
    dataset_size = 10000
    dim = 128
    data = np.random.rand(dataset_size, dim).astype(np.float32)
    
    # Create queries
    num_queries = 50
    queries = np.random.rand(num_queries, dim).astype(np.float32)
    
    # Simulate k-NN search (memory intensive)
    k = 10
    results = []
    for q in queries:
        distances = np.sum((data - q) ** 2, axis=1)
        top_k_indices = np.argsort(distances)[:k]
        results.append(top_k_indices)
    
    return results

def main():
    print("=" * 70)
    print("Cache Miss Monitoring Test")
    print("=" * 70)
    
    monitor = get_cache_monitor()
    
    print(f"\nPerf available: {monitor.perf_available}")
    print(f"Perf command: {monitor.perf_command}")
    
    if not monitor.perf_available:
        print("\n❌ ERROR: perf is not available!")
        print("Please install: sudo apt-get install linux-tools-generic")
        return 1
    
    print("\n" + "-" * 70)
    print("Running test operation with cache miss monitoring...")
    print("-" * 70)
    
    result, metrics = monitor.measure_operation(simulate_anns_operation)
    
    print("\n📊 Cache Miss Metrics:")
    print(f"  Cache References:     {metrics['cache_references']:>15,.0f}")
    print(f"  Cache Misses:         {metrics['cache_misses']:>15,.0f}")
    print(f"  Cache Miss Rate:      {metrics['cache_miss_rate']:>15.2%}")
    print(f"  L1 D-Cache Misses:    {metrics['l1_dcache_misses']:>15,.0f}")
    print(f"  L1 I-Cache Misses:    {metrics['l1_icache_misses']:>15,.0f}")
    print(f"  LLC Misses:           {metrics['llc_misses']:>15,.0f}")
    print(f"  Data TLB Misses:      {metrics['dtlb_misses']:>15,.0f}")
    print(f"  Instr TLB Misses:     {metrics['itlb_misses']:>15,.0f}")
    
    print("\n" + "=" * 70)
    
    if metrics['cache_references'] > 0:
        print("✅ SUCCESS: Cache miss monitoring is working!")
        print("\nYou can now run your congestion experiments:")
        print("  python3 run.py --neurips23track congestion \\")
        print("    --algorithm <algo> --nodocker --rebuild \\")
        print("    --runbook_path <runbook.yaml> --dataset <dataset>")
        return 0
    else:
        print("⚠️  WARNING: All metrics are zero!")
        print("\nPossible issues:")
        print("  1. WSL2 limitations (try with '--wsl-compatible' flag)")
        print("  2. Insufficient permissions (try with sudo)")
        print("  3. Perf events not supported on this system")
        return 1

if __name__ == "__main__":
    sys.exit(main())
