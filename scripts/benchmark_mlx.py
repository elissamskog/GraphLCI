import time
import psutil
import mlx.core as mx
from mlx_lm import load, batch_generate
from typing import List, Dict, Any

# Configuration Constants
MODEL_NAME = "mlx-community/Qwen2.5-14B-Instruct-4bit"
CONTEXT_SIZES = [2048, 4096, 8192, 12288]
BATCH_SIZES = [1, 8, 16, 32]

# Long text for filling context
BASE_TEXT = "The history of the Roman Empire covers the history of ancient Rome from the fall of the Roman Republic in 27 BC until the abdication of Romulus Augustulus in AD 476 in the West, and the Fall of Constantinople in the East in AD 1453. Ancient Rome was a civilization that grew from a small agricultural community founded on the Italian Peninsula in the 9th century BC to a massive empire encompassing most of the Mediterranean and beyond. " * 100

def get_system_metrics() -> Dict[str, float]:
    """Capture current system metrics."""
    process = psutil.Process()
    mem_info = process.memory_info()
    
    # Metal Peak Memory (VRAM equivalent on Unified Memory)
    peak_vram = mx.metal.get_peak_memory() / (1024**3) # GB
    
    return {
        "rss_gb": mem_info.rss / (1024**3),
        "swap_gb": psutil.swap_memory().used / (1024**3),
        "peak_vram_gb": peak_vram
    }

def warmup_model(model, tokenizer):
    """Perform a warm-up run to settle memory."""
    print("Warm-up run...")
    # Use batch_generate for warmup too
    prompt = tokenizer.encode("Hello")
    batch_generate(model, tokenizer, prompts=[prompt], max_tokens=10, verbose=False)
    mx.metal.clear_cache()

def run_benchmark(
    model, 
    tokenizer, 
    num_ctx: int, 
    batch_size: int, 
    max_tokens: int = 100
) -> Dict[str, Any]:
    """Run a single benchmark iteration using batch_generate."""
    
    # Generate prompt to fill context
    target_prompt_tokens = max(100, num_ctx - max_tokens - 100)
    
    base_tokens = tokenizer.encode(BASE_TEXT)
    single_prompt_tokens = []
    while len(single_prompt_tokens) < target_prompt_tokens:
        single_prompt_tokens.extend(base_tokens)
    
    single_prompt_tokens = single_prompt_tokens[:target_prompt_tokens]
    
    # Replicate for batch
    prompts = [single_prompt_tokens] * batch_size
    
    print(f"  > Context: {num_ctx} (Prompt: {len(single_prompt_tokens)} toks), Batch: {batch_size}")

    start_time = time.time()
    
    try:
        response = batch_generate(
            model, 
            tokenizer, 
            prompts=prompts, 
            max_tokens=max_tokens, 
            verbose=False
        )
        # response is BatchResponse
        # It has stats
        tps = response.stats.generation_tps
        
    except Exception as e:
        print(f"Error in batch_generate: {e}")
        tps = 0
        
    end_time = time.time()
    # If using built-in stats, great. If not (e.g. calculation differs), use duration.
    # But batch_generate duration might include prefill.
    # response.stats.generation_tps refers to generation phase usually.
    # Total effective TPS for throughput = total_tokens / total_time
    # Let's see what stats gives us.
    
    metrics = get_system_metrics()
    
    return {
        "num_ctx": num_ctx,
        "batch_size": batch_size,
        "duration": end_time - start_time,
        "tps": tps, # Using MLX reported generation TPS
        "rss_gb": metrics["rss_gb"],
        "peak_vram_gb": metrics["peak_vram_gb"],
        "swap_gb": metrics["swap_gb"],
        "status": "PASS" if metrics["rss_gb"] < 17.5 else "WARN"
    }

def main():
    print("--- MLX Performance Benchmarking (Batch Mode) ---")
    print(f"Model: {MODEL_NAME}")
    
    try:
        model, tokenizer = load(MODEL_NAME)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    warmup_model(model, tokenizer)
    
    # 1. Parameter Sweep
    print("\nStarting Parameter Sweep...")
    print(f"{'Ctx':<6} | {'Batch':<5} | {'TPS':<8} | {'VRAM(GB)':<8} | {'RSS(GB)':<8} | {'Status'}")
    print("-" * 65)
    
    for ctx in CONTEXT_SIZES:
        for batch in BATCH_SIZES:
            # Skip dangerous large inputs
            if ctx >= 8192 and batch >= 16:
                 print(f"{ctx:<6} | {batch:<5} | {'SKIP':<8} | {'-':<8} | {'-':<8} | HIGH RISK")
                 continue
                 
            try:
                res = run_benchmark(model, tokenizer, ctx, batch)
                print(f"{res['num_ctx']:<6} | {res['batch_size']:<5} | {res['tps']:<8.2f} | {res['peak_vram_gb']:<8.2f} | {res['rss_gb']:<8.2f} | {res['status']}")
            except Exception as e:
                 print(f"{ctx:<6} | {batch:<5} | {'ERR':<8} | {'-':<8} | {'-':<8} | FAIL: {e}")
            
            mx.metal.clear_cache()
            time.sleep(1)

    # 2. Parallel Extraction Test
    print("\nRunning Parallel Extraction Test (2 batches sim. parallel workers, ctx=4096)...")
    try:
        # mx.metal.clear_cache() 
        # print(f"Mem usage before parallel: {mx.metal.get_peak_memory()/(1024**3):.2f} GB")
        p_res = run_benchmark(model, tokenizer, num_ctx=4096, batch_size=2)
        print(f"Parallel TPS: {p_res['tps']:.2f}")
        print(f"Memory: VRAM={p_res['peak_vram_gb']:.2f}GB, RSS={p_res['rss_gb']:.2f}GB")
    except Exception as e:
        print(f"Parallel test failed: {e}")
    
    print("\n--- Benchmark Complete ---")

if __name__ == "__main__":
    main()
