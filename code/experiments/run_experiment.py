"""Batched causal inference. Failed execution requests never silently fall back."""
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from metrics.energy import HybridEnergyMonitor


def load_model_with_fallback(model_name, quantization="fp32", device="cpu"):
    """Historical API name; unsupported precision/device requests now raise."""
    kwargs = {"low_cpu_mem_usage": True}

    if quantization == "fp32":
        kwargs["torch_dtype"] = torch.float32

    elif quantization in ("int8", "int4"):
        if device != "cuda":
            raise ValueError(
                "INT8/INT4 require CUDA here; CPU low-bit backends are not validated"
            )

        options = (
            {"load_in_8bit": True}
            if quantization == "int8"
            else {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            }
        )

        kwargs.update(
            quantization_config=BitsAndBytesConfig(**options),
            device_map={"": 0},
        )
    else:
        raise ValueError(f"Unknown precision: {quantization}")

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if quantization == "fp32":
        model = model.to(device)
    elif not getattr(
        model,
        "is_loaded_in_8bit" if quantization == "int8" else "is_loaded_in_4bit",
        False,
    ):
        raise RuntimeError("Requested quantization was not applied")

    return model


def count_output_tokens(rows, eos_token_id):
    """Include the first EOS, exclude padding added after it in a batch."""
    eos_ids = set(
        eos_token_id
        if isinstance(eos_token_id, (list, tuple))
        else [eos_token_id]
    )

    total = 0
    for row in rows:
        for token in row:
            total += 1
            if token in eos_ids:
                break

    return total


def run_single_experiment(
    config,
    model_name,
    prompts,
    batch_size,
    quantization="fp32",
    measurement_method="software",
    hardware="cpu",
):
    metadata = dict(
        model=model_name,
        quantization=quantization,
        batch_size=batch_size,
        hardware=hardware,
        measurement_method=measurement_method,
    )

    monitor = None
    started = False

    try:
        if not prompts or batch_size < 1 or config["max_new_tokens"] < 1:
            raise ValueError(
                "Nonempty prompts, positive batch size and token limit required"
            )

        if hardware not in ("cpu", "gpu"):
            raise ValueError(f"Unknown hardware: {hardware}")

        if hardware == "gpu" and not torch.cuda.is_available():
            raise ValueError("Requested GPU is unavailable")

        device = "cuda" if hardware == "gpu" else "cpu"

        if measurement_method == "hardware" and device != "cuda":
            raise ValueError("NVML GPU energy cannot measure a CPU experiment")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = "left"

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = load_model_with_fallback(model_name, quantization, device)
        model.eval()

        metadata.update(
            actual_device=str(next(model.parameters()).device),
            actual_precision=quantization,
            model_params=sum(p.numel() for p in model.parameters()),
        )
        metadata["model_params_m"] = metadata["model_params"] / 1e6

        def sync():
            if device == "cuda":
                torch.cuda.synchronize()

        batches = [
            tokenizer(
                prompts[i:i + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.get("max_input_tokens", 512),
            )
            for i in range(0, len(prompts), batch_size)
        ]

        generation = dict(
            max_new_tokens=config["max_new_tokens"],
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        with torch.inference_mode():
            for _ in range(config.get("warmup_runs", 1)):
                model.generate(
                    **{k: v.to(device) for k, v in batches[0].items()},
                    **generation,
                )

        sync()

        monitor = HybridEnergyMonitor(
            method=measurement_method,
            carbon_intensity=config.get("carbon_intensity", 0.233),
        )

        prefill_times = []
        generation_times = []
        input_tokens = 0
        output_tokens = 0

        monitor.start()
        started = True
        start = time.perf_counter()

        for batch in batches:
            inputs = {k: v.to(device) for k, v in batch.items()}
            input_tokens += int(inputs["attention_mask"].sum().item())
            phase = {}

            def before_forward(module, args):
                if "start" not in phase:
                    sync()
                    phase["start"] = time.perf_counter()

            def after_forward(module, args, output):
                if "elapsed" not in phase:
                    sync()
                    phase["elapsed"] = time.perf_counter() - phase["start"]

            pre = model.register_forward_pre_hook(before_forward)
            post = model.register_forward_hook(after_forward)

            try:
                sync()
                batch_start = time.perf_counter()

                with torch.inference_mode():
                    generated = model.generate(**inputs, **generation)

                sync()
                generation_times.append(time.perf_counter() - batch_start)
            finally:
                pre.remove()
                post.remove()

            if "elapsed" not in phase:
                raise RuntimeError(
                    "Could not observe generation's prefill forward pass"
                )

            prefill_times.append(phase["elapsed"])

            suffix = generated[:, inputs["input_ids"].shape[1]:].tolist()
            output_tokens += count_output_tokens(
                suffix,
                tokenizer.eos_token_id,
            )

        sync()
        total_time = time.perf_counter() - start

        started = False
        energy = monitor.stop()

        if output_tokens <= 0:
            raise RuntimeError("No output tokens generated")

        prefill = sum(prefill_times)
        decode = sum(generation_times) - prefill

        return dict(
            metadata,
            **energy,
            status="ok",
            error="",
            num_queries=len(prompts),
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_time=total_time,
            throughput=output_tokens / total_time,
            energy_per_token=energy["energy_kwh"] / output_tokens,
            energy_per_query=energy["energy_kwh"] / len(prompts),
            carbon_per_query=energy["carbon_kg"] / len(prompts),
            batch_latency_mean=sum(generation_times) / len(batches),
            num_batches=len(batches),
            effective_batch_size=min(batch_size, len(prompts)),
            prefill_time_total=prefill,
            decode_time_total=decode,
            prefill_time_mean=prefill / len(batches),
            decode_time_mean=decode / len(batches),
            phase_ratio=decode / prefill if prefill else None,
            timing_boundary=(
                "prefill=first forward; "
                "decode=generation remainder including orchestration"
            ),
        )

    except Exception as exc:
        return dict(
            metadata,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )

    finally:
        if started:
            try:
                monitor.stop()
            except Exception as cleanup_error:
                import warnings
                warnings.warn(
                    f"Energy monitor cleanup failed: {cleanup_error}"
                )