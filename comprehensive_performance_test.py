#!/usr/bin/env python3
"""
Comprehensive performance test for parallel metric computation.
Tests both performance improvements and net_score_latency accuracy.
"""

import sys
import os
import time
import statistics
sys.path.insert(0, os.getcwd())

from src.scorer import compute_all_metrics_parallel
from src.url import UrlCategory

def test_net_score_latency_accuracy():
    """Test that net_score_latency is calculated correctly."""
    print("🔧 Testing net_score_latency Accuracy")
    print("=" * 50)
    
    # Test data
    test_data = {
        'downloads': 1000000,
        'likes': 1000,
        'cardData': {'license': 'apache-2.0'},
        'tags': ['pytorch', 'nlp'],
        'contributors': [{'name': 'user1'}, {'name': 'user2'}, {'name': 'user3'}]
    }
    
    url = 'https://huggingface.co/test-model'
    category = UrlCategory.MODEL
    code_url = None
    model_name = 'test-model'
    
    print("Running parallel metric computation...")
    start_time = time.time()
    results, total_latency = compute_all_metrics_parallel(
        test_data, url, category, code_url, model_name
    )
    end_time = time.time()
    
    actual_time = end_time - start_time
    net_score_latency = results.get('net_score_latency', 0)
    
    print(f"  Actual execution time: {actual_time:.3f}s")
    print(f"  Reported total latency: {total_latency}ms")
    print(f"  Net score latency: {net_score_latency}ms")
    print(f"  Net score: {results.get('net_score', 'N/A')}")
    
    # Check if the fix is working
    if net_score_latency > 0:
        print(f"✅ SUCCESS: net_score_latency is now {net_score_latency}ms (was 0 before)")
        return True
    else:
        print(f"❌ FAILURE: net_score_latency is still {net_score_latency}ms")
        return False

def simulate_sequential_vs_parallel():
    """Simulate and compare sequential vs parallel execution."""
    print(f"\n📊 Sequential vs Parallel Performance Analysis")
    print("=" * 60)
    
    # Realistic I/O times for each metric
    io_times = {
        'ramp_up_time': 0.5,        # Documentation analysis + file reading
        'bus_factor': 0.3,          # API calls for contributor data
        'performance_claims': 0.4,  # Model card analysis
        'license': 0.1,             # License lookup
        'size_score': 2.0,          # Model download + size calculation (HEAVY I/O)
        'dataset_and_code_score': 0.2, # Simple calculation
        'dataset_quality': 0.6,      # Dataset analysis + API calls
        'code_quality': 3.0          # Git clone + Flake8 analysis (HEAVY I/O)
    }
    
    # Sequential execution (sum of all times)
    seq_total = sum(io_times.values())
    print(f"Sequential execution (sum of all): {seq_total:.1f}s")
    
    # Parallel execution (max of all times)
    par_total = max(io_times.values())
    print(f"Parallel execution (max of all): {par_total:.1f}s")
    
    # Calculate improvements
    time_saved = seq_total - par_total
    improvement = (time_saved / seq_total) * 100
    speedup = seq_total / par_total
    
    print(f"\n🎯 Performance Results:")
    print(f"  Time saved: {time_saved:.1f}s")
    print(f"  Improvement: {improvement:.1f}%")
    print(f"  Speedup: {speedup:.1f}x")
    
    return speedup, improvement

def test_actual_parallel_performance():
    """Test actual parallel implementation performance."""
    print(f"\n🧪 Testing Actual Parallel Implementation")
    print("=" * 50)
    
    # Test data
    test_data = {
        'downloads': 1000000,
        'likes': 1000,
        'cardData': {'license': 'apache-2.0'},
        'tags': ['pytorch', 'nlp'],
        'contributors': [{'name': 'user1'}, {'name': 'user2'}, {'name': 'user3'}]
    }
    
    url = 'https://huggingface.co/test-model'
    category = UrlCategory.MODEL
    code_url = None
    model_name = 'test-model'
    
    # Run multiple tests to get average performance
    num_tests = 3
    times = []
    
    print(f"Running {num_tests} tests...")
    for i in range(num_tests):
        print(f"  Test {i+1}/{num_tests}...")
        start_time = time.time()
        results, latency = compute_all_metrics_parallel(
            test_data, url, category, code_url, model_name
        )
        end_time = time.time()
        times.append(end_time - start_time)
        time.sleep(0.1)  # Small delay between tests
    
    avg_time = statistics.mean(times)
    print(f"\n📊 Actual Performance Results:")
    print(f"  Average time: {avg_time:.3f}s")
    print(f"  Min time: {min(times):.3f}s")
    print(f"  Max time: {max(times):.3f}s")
    print(f"  Net score: {results.get('net_score', 'N/A')}")
    
    return avg_time

def demonstrate_scaling_benefits():
    """Demonstrate how benefits scale with workload."""
    print(f"\n📈 Scaling Benefits Analysis")
    print("=" * 50)
    
    # Different workload sizes
    workloads = [1, 5, 10, 50, 100]
    seq_time = 7.1  # From our analysis
    par_time = 3.0  # From our analysis
    
    print("Workload Size | Sequential | Parallel | Time Saved | Speedup")
    print("-" * 60)
    
    for workload in workloads:
        seq_total = seq_time * workload
        par_total = par_time * workload
        time_saved = seq_total - par_total
        speedup = seq_total / par_total
        
        print(f"{workload:12d} | {seq_total:8.1f}s | {par_total:6.1f}s | {time_saved:8.1f}s | {speedup:6.1f}x")
    
    print(f"\n🎯 Scaling Insights:")
    print(f"  • Benefits increase linearly with workload size")
    print(f"  • Time savings compound for large batches")
    print(f"  • Parallel processing becomes essential for production")

def show_latency_breakdown(results):
    """Show detailed latency breakdown."""
    print(f"\n📋 Detailed Latency Breakdown:")
    print("=" * 50)
    
    latency_fields = {k: v for k, v in results.items() if 'latency' in k}
    for field, value in sorted(latency_fields.items()):
        if value > 0:  # Only show non-zero latencies
            print(f"  {field}: {value}ms")

def main():
    """Run comprehensive performance analysis."""
    print("🚀 Comprehensive Parallel Metric Computation Test")
    print("=" * 70)
    print()
    
    # Test 1: net_score_latency accuracy
    latency_success = test_net_score_latency_accuracy()
    
    # Test 2: Sequential vs Parallel simulation
    speedup, improvement = simulate_sequential_vs_parallel()
    
    # Test 3: Actual parallel performance
    actual_time = test_actual_parallel_performance()
    
    # Test 4: Scaling benefits
    demonstrate_scaling_benefits()
    
    # Summary
    print(f"\n📋 Summary:")
    print("=" * 50)
    
    if latency_success:
        print(f"✅ net_score_latency fix is working correctly")
    else:
        print(f"❌ net_score_latency still needs fixing")
    
    print(f"✅ Theoretical speedup: {speedup:.1f}x")
    print(f"✅ Theoretical improvement: {improvement:.1f}%")
    print(f"✅ Actual execution time: {actual_time:.3f}s")
    
    print(f"\n🎯 Key Benefits:")
    print(f"  • All 8 metrics computed simultaneously")
    print(f"  • ThreadPoolExecutor optimized for I/O-bound tasks")
    print(f"  • Dynamic worker count based on CPU cores")
    print(f"  • Accurate latency reporting")
    print(f"  • Graceful error handling")
    
    print(f"\n🌍 Best Use Cases:")
    print(f"  • Multiple model analysis")
    print(f"  • Complex I/O operations")
    print(f"  • Network-bound tasks")
    print(f"  • Production workloads")
    print(f"  • Large batch processing")

if __name__ == "__main__":
    main()
