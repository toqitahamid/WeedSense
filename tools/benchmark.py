"""
Benchmark script for WeedSense

Measures:
- FPS (frames per second)
- FLOPs (floating point operations)
- Parameters
- GPU memory usage

Usage:
    python tools/benchmark.py \
        --checkpoint pretrained/weedsense_iccv2025.pth \
        --input-size 512 512 \
        --batch-size 1 \
        --warmup 100 \
        --iters 300
"""

import argparse
import time
import numpy as np
import torch
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from weedsense import WeedSense


def preheat_gpu(batch_size, input_size):
    """Pre-heat GPU to stabilize thermal state"""
    print("Pre-heating GPU...")
    dummy = torch.randn(batch_size, 3, input_size[0], input_size[1]).cuda()
    with torch.no_grad():
        for _ in tqdm(range(300), desc="Pre-heating"):
            dummy = dummy * dummy
            dummy = torch.clamp(dummy, -1, 1)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print("Pre-heating completed\n")


def run_benchmark(model, input_generator, batch_size, warmup, iters):
    """Run benchmark and return metrics"""
    # Warmup
    torch.cuda.synchronize()
    for _ in tqdm(range(warmup), desc='Warmup'):
        with torch.no_grad():
            _ = model(input_generator())
        torch.cuda.synchronize()
    
    # Benchmark
    timings = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    torch.cuda.synchronize()
    for _ in tqdm(range(iters), desc='Benchmarking'):
        inp = input_generator()
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            _ = model(inp)
        
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    
    # Memory measurement
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_generator())
            torch.cuda.synchronize()
    memory = torch.cuda.max_memory_allocated() / 1024**2
    
    # Calculate FPS
    fps_values = batch_size / np.array(timings)
    mean_fps = np.mean(fps_values)
    std_fps = np.std(fps_values)
    
    return {
        'fps': mean_fps,
        'fps_std': std_fps,
        'memory': memory,
        'timings': timings
    }


def main():
    parser = argparse.ArgumentParser(description='WeedSense Benchmark')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    parser.add_argument('--num-classes', type=int, default=17, help='Number of classes')
    parser.add_argument('--num-weeks', type=int, default=11, help='Number of growth stages')
    parser.add_argument('--input-size', nargs=2, type=int, default=[512, 512], help='Input size [H W]')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--warmup', type=int, default=100, help='Warmup iterations')
    parser.add_argument('--iters', type=int, default=300, help='Benchmark iterations')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs')
    parser.add_argument('--preheat', action='store_true', help='Pre-heat GPU')
    args = parser.parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("Error: CUDA not available!")
        sys.exit(1)
    
    # Print configuration
    print(f"\n{'='*60}")
    print("WeedSense Benchmark Configuration")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Classes: {args.num_classes}, Weeks: {args.num_weeks}")
    print(f"Input size: {args.input_size[0]}x{args.input_size[1]}")
    print(f"Batch size: {args.batch_size}")
    print(f"Warmup: {args.warmup}, Benchmark: {args.iters}, Runs: {args.runs}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")
    
    # Pre-heat GPU
    if args.preheat:
        preheat_gpu(args.batch_size, args.input_size)
    
    # Load model
    print("Loading model...")
    model = WeedSense(num_classes=args.num_classes, num_weeks=args.num_weeks)
    model.load_pretrained(args.checkpoint)
    model = model.cuda().eval()
    print("Model loaded\n")
    
    # Input generator
    H, W = args.input_size
    input_generator = lambda: torch.randn(args.batch_size, 3, H, W, device='cuda')
    
    # Calculate FLOPs and parameters
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_generator(),), verbose=False)
        flops_giga = flops / 1e9
        params_millions = params / 1e6
    except ImportError:
        print("Warning: thop not installed. Install with: pip install thop")
        flops_giga = None
        params_millions = model.get_num_params()
    
    # Run multiple benchmark passes
    all_fps = []
    all_memory = []
    
    for run in range(args.runs):
        print(f"{'='*60}")
        print(f"Benchmark Run {run+1}/{args.runs}")
        print(f"{'='*60}\n")
        
        results = run_benchmark(model, input_generator, args.batch_size, args.warmup, args.iters)
        all_fps.append(results['fps'])
        all_memory.append(results['memory'])
        
        print(f"\nRun {run+1} Results:")
        print(f"  FPS: {results['fps']:.2f} ± {results['fps_std']:.2f}")
        print(f"  Memory: {results['memory']:.1f} MB\n")
    
    # Final results
    mean_fps = np.mean(all_fps)
    std_fps = np.std(all_fps)
    mean_memory = np.mean(all_memory)
    
    print(f"\n{'='*60}")
    print("Final Benchmark Results")
    print(f"{'='*60}")
    print(f"Model: WeedSense")
    print(f"  Classes: {args.num_classes}, Weeks: {args.num_weeks}")
    print(f"  Parameters: {params_millions:.2f} M")
    if flops_giga is not None:
        print(f"  FLOPs: {flops_giga:.2f} G")
    print(f"\nPerformance:")
    print(f"  FPS: {mean_fps:.2f} ± {std_fps:.2f} (across {args.runs} runs)")
    print(f"  Latency: {1000/mean_fps:.2f} ms/image")
    print(f"  GPU Memory: {mean_memory:.1f} MB")
    print(f"  Input Size: {args.input_size[0]}x{args.input_size[1]}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

