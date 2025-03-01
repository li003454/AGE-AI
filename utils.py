import time

# Calculate the time for each function
performance_metrics = {}

def profile_func(func):
    """A profiling decorator that records execution time and stores it in performance_metrics."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        #print(f"[PROFILE] {func.__name__} executed in {elapsed:.4f} seconds")
        # 记录性能数据
        if func.__name__ not in performance_metrics:
            performance_metrics[func.__name__] = []
        performance_metrics[func.__name__].append(elapsed)
        return result
    return wrapper

def print_performance_metrics():
    """Print a summary of performance metrics (call counts and average execution time)."""
    print("\n=== Performance Metrics Summary ===")
    for func_name, times in performance_metrics.items():
        avg_time = sum(times) / len(times)
        call_count = len(times)
        print(f"{func_name}: called {call_count} times, average time: {avg_time:.4f} seconds")
    print("===================================\n")
