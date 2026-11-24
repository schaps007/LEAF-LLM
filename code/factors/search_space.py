"""
Search space for OFAT experiments - With LIGHTWEIGHT models only
"""

import torch

# ============================================================
# MODEL LIST - ONLY LIGHTWEIGHT MODELS (< 500M parameters)
# ============================================================
MODEL_LIST = [
    # Current models (tested and working)
    "gpt2",                    # 124M - ✅ Works
    "distilgpt2",              # 82M - ✅ Works
    "facebook/opt-125m",       # 125M - ✅ Works
    "facebook/opt-350m",       # 350M - ✅ Works
    
    # COMMENT OUT large models that cause issues
    # "EleutherAI/gpt-neo-125M", # 125M - Remove (duplicate)
    # "EleutherAI/gpt-neo-1.3B", # 1.3B - Too slow
    # "microsoft/phi-2",         # 2.7B - Too large, causing hangs
    # "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B - Too large
    # "google/gemma-2b",                    # 2B - Too large
]

# ============================================================
# QUANTIZATION LEVELS
# ============================================================
QUANTIZATION_LEVELS = [
    "fp32",
    "int8",  # Temporarily disable if causing issues
    "int4"  # Temporarily disable if causing issues
]

# ============================================================
# BATCH SIZES
# ============================================================
BATCH_SIZES = [1, 4]

# ============================================================
# WORKLOAD TYPES
# ============================================================
WORKLOAD_TYPES = [
    "short",
    "medium",
    "long"
]

# ============================================================
# HARDWARE PLATFORMS
# ============================================================
HARDWARE_PLATFORMS = [
    "cpu",
    "gpu",
]

# ============================================================
# MEASUREMENT METHODS
# ============================================================
MEASUREMENT_METHODS = [
    "software",
    # "hardware",  # Uncomment if you have NVIDIA GPU
]

# ============================================================
# BASELINE CONFIGURATION
# ============================================================
BASELINE_CONFIG = {
    "model": "gpt2",
    "quantization": "fp32",
    "batch_size": 1,
    "workload": "short",
    "hardware": "cpu",
    "measurement_method": "software"
}

def get_ofat_experiments():
    """
    Generate OFAT experiments covering ALL factors
    """
    baseline = BASELINE_CONFIG
    experiments = []
    
    # 1. Model comparison
    for model in MODEL_LIST:
        if model != baseline["model"]:
            experiments.append({
                "factor": "model",
                "model": model,
                "quantization": baseline["quantization"],
                "batch_size": baseline["batch_size"],
                "workload": baseline["workload"],
                "hardware": baseline["hardware"],
                "measurement_method": baseline["measurement_method"],
                "description": f"Model: {model.split('/')[-1]}"
            })
    
    # 2. Quantization test
    for quant in QUANTIZATION_LEVELS:
        if quant != baseline["quantization"]:
            experiments.append({
                "factor": "quantization",
                "model": baseline["model"],
                "quantization": quant,
                "batch_size": baseline["batch_size"],
                "workload": baseline["workload"],
                "hardware": baseline["hardware"],
                "measurement_method": baseline["measurement_method"],
                "description": f"Quantization: {quant}"
            })
    
    # 3. Batch size test
    for batch in BATCH_SIZES:
        if batch != baseline["batch_size"]:
            experiments.append({
                "factor": "batch_size",
                "model": baseline["model"],
                "quantization": baseline["quantization"],
                "batch_size": batch,
                "workload": baseline["workload"],
                "hardware": baseline["hardware"],
                "measurement_method": baseline["measurement_method"],
                "description": f"Batch Size: {batch}"
            })
    
    # 4. Workload test
    for workload in WORKLOAD_TYPES:
        if workload != baseline["workload"]:
            experiments.append({
                "factor": "workload",
                "model": baseline["model"],
                "quantization": baseline["quantization"],
                "batch_size": baseline["batch_size"],
                "workload": workload,
                "hardware": baseline["hardware"],
                "measurement_method": baseline["measurement_method"],
                "description": f"Workload: {workload}"
            })
    
    # 5. Hardware platform test (if GPU available)
    if torch.cuda.is_available():
        for hardware in HARDWARE_PLATFORMS:
            if hardware != baseline["hardware"]:
                experiments.append({
                    "factor": "hardware",
                    "model": baseline["model"],
                    "quantization": baseline["quantization"],
                    "batch_size": baseline["batch_size"],
                    "workload": baseline["workload"],
                    "hardware": hardware,
                    "measurement_method": baseline["measurement_method"],
                    "description": f"Hardware: {hardware}"
                })
    else:
        print("⚠️  No GPU detected - skipping hardware platform tests")
    
    # 6. Measurement method test (if hardware measurement available)
    try:
        import pynvml
        pynvml.nvmlInit()
        for method in MEASUREMENT_METHODS:
            if method != baseline["measurement_method"]:
                experiments.append({
                    "factor": "measurement_method",
                    "model": baseline["model"],
                    "quantization": baseline["quantization"],
                    "batch_size": baseline["batch_size"],
                    "workload": baseline["workload"],
                    "hardware": baseline["hardware"],
                    "measurement_method": method,
                    "description": f"Measurement: {method}"
                })
    except:
        print("⚠️  NVIDIA NVML not available - skipping measurement method tests")
    
    return experiments

def get_experiment_groups():
    """Group experiments by factor"""
    groups = {
        "baseline": BASELINE_CONFIG,
        "model": [],
        "quantization": [],
        "batch_size": [],
        "workload": [],
        "hardware": [],
        "measurement_method": []
    }
    
    for exp in get_ofat_experiments():
        groups[exp["factor"]].append(exp)
    
    return groups

def get_model_info(model_name):
    """Get model information"""
    model_info = {
        "gpt2": {"params": "124M", "type": "decoder-only"},
        "distilgpt2": {"params": "82M", "type": "decoder-only"},
        "facebook/opt-125m": {"params": "125M", "type": "decoder-only"},
        "facebook/opt-350m": {"params": "350M", "type": "decoder-only"},
        "EleutherAI/gpt-neo-125M": {"params": "125M", "type": "decoder-only"},
        "EleutherAI/gpt-neo-1.3B": {"params": "1.3B", "type": "decoder-only"},
        "microsoft/phi-2": {"params": "2.7B", "type": "decoder-only"},
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {"params": "1.1B", "type": "decoder-only"},
        "google/gemma-2b": {"params": "2B", "type": "decoder-only"},
    }
    return model_info.get(model_name, {"params": "unknown", "type": "unknown"})