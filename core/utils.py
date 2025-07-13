import torch
import time
from typing import Callable, Dict, Tuple
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import psutil

NUM_WARMUP_ITERATIONS = 100

def cuda_timer(func):
    def wrapper(*args, **kwargs):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record(stream=torch.cuda.current_stream())
        result = func(*args, **kwargs)
        end_time.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        return result, start_time.elapsed_time(end_time)
    return wrapper

def cpu_timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time() - start_time
        return result, end_time * 1000
    return wrapper

def gpu_mem_usage(func):
    def wrapper(*args, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        allocated_memory = torch.cuda.max_memory_allocated()
        result = func(*args, **kwargs)
        return result, (torch.cuda.max_memory_allocated() - allocated_memory) / 2 ** 20
    return wrapper

def cpu_mem_usage(func):
    def wrapper(*args, **kwargs):
        allocated_memory = psutil.Process().memory_info().rss
        result = func(*args, **kwargs)
        return result, (psutil.Process().memory_info().rss - allocated_memory) / 2 ** 20
    return wrapper

def run_test(
    model_wrapper: Callable,
    data_preprocess: Callable = None,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    num_runs: int = 1000,
    min_batch_size: int = 1,
    max_batch_size: int = 1,
    batch_step: int = 1,
    dataset: Dataset = None,
    timer_type: str = 'cuda'
) -> Dict[Tuple[int, int, int], float]:
    shapes = [(size, *input_shape) for size in range(min_batch_size, max_batch_size + 1, batch_step)]
    results = {}
    timer = cuda_timer if timer_type == 'cuda' else cpu_timer
    for shape in shapes:
        dataloader = DataLoader(dataset, batch_size=shape[0], shuffle=False, drop_last=True)
        with torch.no_grad():
            for _ in tqdm(range(NUM_WARMUP_ITERATIONS), desc=f'Warmup for shape {shape}'):
                dummy_input = torch.randn(shape, device=timer_type)
                if data_preprocess:
                    dummy_input = data_preprocess(dummy_input)
                model_wrapper(dummy_input)
            times = []
            for _ in range(num_runs):
                for batch in tqdm(dataloader, desc=f'Testing for shape {shape}, iter {_}'):
                    image = batch[0].to(timer_type)
                    if data_preprocess:
                        image = data_preprocess(image)
                    result, time = timer(model_wrapper)(image)
                    times.append(time)
        times = np.array(times)
        times = times[~np.isnan(times)]
        times = times[times < np.percentile(times, 90)]
        times = times[times > np.percentile(times, 10)]
        results[shape] = np.mean(times).item()
    return results