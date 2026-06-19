from __future__ import annotations

"""Stage-A rollout sampling from the SFT policy for RM data construction.

This launcher mirrors verl's rollout placement rule on a single node:

    rollout_instance_count = n_gpus_per_node // tensor_parallel_size

Each rollout instance runs in its own process and owns exactly one GPU group.
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from common import load_records, normalize_prompt, preview_records, save_records_as_parquet

_INTERNAL_HELP = argparse.SUPPRESS
_DEFAULT_FALLBACK_GPU_MEMORY_UTILIZATION = 0.7
_DEFAULT_FALLBACK_MAX_NUM_SEQS = 256


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for rollout generation from the SFT policy."""
    parser = argparse.ArgumentParser(description="Run multi-process vLLM rollout generation for RM data construction.")
    parser.add_argument("--data", type=str, default=None, help="Input RL dataset path or Hugging Face dataset name.")
    parser.add_argument("--dataset-split", type=str, default=None, help="Optional dataset split.")
    parser.add_argument("--prompt-key", type=str, default="prompt", help="Prompt column name.")
    parser.add_argument("--begin", type=int, default=0, help="Inclusive begin index for slicing the dataset.")
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Exclusive end index for slicing the dataset. Defaults to the dataset length.",
    )
    parser.add_argument("--output-path", type=Path, default=None, help="Output dataset path (.json, .jsonl, or .parquet).")
    parser.add_argument("--model-name-or-path", type=str, default=None, help="SFT model path or id used for rollout.")
    parser.add_argument("--tokenizer-name-or-path", type=str, default=None, help="Optional tokenizer path.")
    parser.add_argument("--enable-thinking", type=bool, default=False, help="Optional thinking mode.", action=argparse.BooleanOptionalAction)
    parser.add_argument("--num-samples", type=int, default=1, help="Number of responses sampled per prompt.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling threshold.")
    parser.add_argument("--top-k", type=int, default=-1, help="Top-k sampling threshold.")
    parser.add_argument("--min-p", type=float, default=0.0, help="Min-p sampling threshold.")
    parser.add_argument("--max-prompt-length", type=int, default=2048, help="Prompt truncation length in tokens.")
    parser.add_argument("--max-response-tokens", type=int, default=2048, help="Maximum new tokens for each sample.")
    parser.add_argument(
        "--preprocess-num-proc",
        type=int,
        default=max(1, min(16, os.cpu_count() or 1)),
        help="Number of worker processes used by dataset.map for prompt preprocessing.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help=(
            "Number of GPUs per vLLM instance. Defaults to 1, "
            "which preserves the old single-instance behavior."
        ),
    )
    parser.add_argument(
        "--n-gpus-per-node",
        type=int,
        default=None,
        help="Number of GPUs reserved for rollout on this node. Defaults to all visible GPUs.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Optional comma-separated GPU ids to use, for example '0,1,2,3'.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=_DEFAULT_FALLBACK_GPU_MEMORY_UTILIZATION,
        help="vLLM gpu_memory_utilization. Defaults to a conservative startup value.",
    )
    parser.add_argument("--max-num-seqs", type=int, default=_DEFAULT_FALLBACK_MAX_NUM_SEQS, help="vLLM max_num_seqs.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow remote code for model and tokenizer.")
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable torch.compile and CUDA graphs in vLLM to reduce startup memory usage. Enabled by default for safer startup.",
    )
    parser.add_argument(
        "--disable-init-eager-fallback",
        action="store_true",
        help=(
            "Do not retry vLLM initialization with conservative eager-mode settings "
            "after the first startup failure."
        ),
    )

    parser.add_argument("--worker-mode", action="store_true", help=_INTERNAL_HELP)
    parser.add_argument("--worker-index", type=int, default=None, help=_INTERNAL_HELP)
    parser.add_argument("--worker-input-path", type=Path, default=None, help=_INTERNAL_HELP)
    parser.add_argument("--worker-output-path", type=Path, default=None, help=_INTERNAL_HELP)
    parser.add_argument("--worker-gpu-ids", type=str, default=None, help=_INTERNAL_HELP)

    args = parser.parse_args()
    validate_args(parser, args)
    return args


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate CLI arguments for either coordinator mode or worker mode."""
    required_fields = {
        "model_name_or_path": args.model_name_or_path,
    }
    if args.worker_mode:
        required_fields.update(
            {
                "worker_input_path": args.worker_input_path,
                "worker_output_path": args.worker_output_path,
                "worker_gpu_ids": args.worker_gpu_ids,
            }
        )
    else:
        required_fields.update(
            {
                "data": args.data,
                "output_path": args.output_path,
            }
        )

    missing_args = [f"--{name.replace('_', '-')}" for name, value in required_fields.items() if value is None]
    if missing_args:
        parser.error(f"Missing required arguments: {', '.join(missing_args)}")

    positive_int_fields = {
        "num_samples": args.num_samples,
        "max_prompt_length": args.max_prompt_length,
        "max_response_tokens": args.max_response_tokens,
        "preprocess_num_proc": args.preprocess_num_proc,
        "max_num_seqs": args.max_num_seqs,
    }
    if args.tensor_parallel_size is not None:
        positive_int_fields["tensor_parallel_size"] = args.tensor_parallel_size
    if args.n_gpus_per_node is not None:
        positive_int_fields["n_gpus_per_node"] = args.n_gpus_per_node
    if args.worker_index is not None:
        positive_int_fields["worker_index"] = args.worker_index + 1
    for field_name, value in positive_int_fields.items():
        if value <= 0:
            parser.error(f"--{field_name.replace('_', '-')} must be greater than 0")

    if args.begin < 0:
        parser.error("--begin must be greater than or equal to 0")
    if args.end is not None and args.end < 0:
        parser.error("--end must be greater than or equal to 0")
    if args.end is not None and args.begin > args.end:
        parser.error("--begin must be less than or equal to --end")
    if args.gpu_memory_utilization <= 0 or args.gpu_memory_utilization > 1:
        parser.error("--gpu-memory-utilization must be in the range (0, 1]")


def parse_gpu_id_list(gpu_ids: str) -> list[str]:
    """Parse a comma-separated GPU id list while keeping ids as strings."""
    tokens = [token.strip() for token in gpu_ids.split(",") if token.strip()]
    if not tokens:
        raise ValueError("GPU id list cannot be empty.")
    if len(set(tokens)) != len(tokens):
        raise ValueError(f"GPU ids must be unique, got: {gpu_ids}")
    return tokens


def discover_visible_gpu_ids() -> list[str]:
    """Discover the visible GPU identifiers for the current process."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        return parse_gpu_id_list(visible_devices)

    import torch

    gpu_count = torch.cuda.device_count()
    if gpu_count <= 0:
        raise RuntimeError("No CUDA GPUs are available for rollout generation.")
    return [str(index) for index in range(gpu_count)]


def resolve_launch_layout(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve GPU grouping using the same TP/DP relationship as verl rollout."""
    if args.gpu_ids:
        selected_gpu_ids = parse_gpu_id_list(args.gpu_ids)
        if args.n_gpus_per_node is not None and args.n_gpus_per_node != len(selected_gpu_ids):
            raise ValueError(
                "--n-gpus-per-node must match the number of ids in --gpu-ids when both are provided."
            )
    else:
        visible_gpu_ids = discover_visible_gpu_ids()
        requested_gpu_count = args.n_gpus_per_node or len(visible_gpu_ids)
        if requested_gpu_count > len(visible_gpu_ids):
            raise ValueError(
                f"Requested {requested_gpu_count} GPUs but only {len(visible_gpu_ids)} are visible: {visible_gpu_ids}"
            )
        selected_gpu_ids = visible_gpu_ids[:requested_gpu_count]

    n_gpus_per_node = len(selected_gpu_ids)
    tensor_parallel_size = args.tensor_parallel_size or n_gpus_per_node
    if tensor_parallel_size > n_gpus_per_node:
        raise ValueError(
            f"tensor_parallel_size={tensor_parallel_size} cannot exceed n_gpus_per_node={n_gpus_per_node}"
        )
    if n_gpus_per_node % tensor_parallel_size != 0:
        raise ValueError(
            f"n_gpus_per_node={n_gpus_per_node} must be divisible by tensor_parallel_size={tensor_parallel_size}"
        )

    gpu_groups = [
        selected_gpu_ids[offset : offset + tensor_parallel_size]
        for offset in range(0, n_gpus_per_node, tensor_parallel_size)
    ]
    return {
        "selected_gpu_ids": selected_gpu_ids,
        "n_gpus_per_node": n_gpus_per_node,
        "tensor_parallel_size": tensor_parallel_size,
        "rollout_instance_count": len(gpu_groups),
        "gpu_groups": gpu_groups,
    }


def truncate_prompt(tokenizer, rendered_prompt: str, max_prompt_length: int) -> str:
    """Trim the rendered prompt to the context budget before sampling rollouts."""
    prompt_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
    if len(prompt_ids) <= max_prompt_length:
        return rendered_prompt
    trimmed_ids = prompt_ids[-max_prompt_length:]
    return tokenizer.decode(trimmed_ids)


def preprocess_prompts(args: argparse.Namespace, raw_records: list[dict[str, Any]]) -> tuple[list[Any], list[str]]:
    """Normalize prompts and render them into strings once in the coordinator."""
    from datasets import Dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path or args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    preprocess_dataset = Dataset.from_list(raw_records)

    def preprocess_record(record: dict[str, Any]) -> dict[str, Any]:
        prompt_messages = normalize_prompt(record[args.prompt_key])
        rendered_prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=args.enable_thinking)
        if tokenizer.bos_token and rendered_prompt.startswith(tokenizer.bos_token):
            rendered_prompt = rendered_prompt[len(tokenizer.bos_token) :]
        return {
            "_normalized_prompt": prompt_messages,
            "_rendered_prompt": truncate_prompt(tokenizer, rendered_prompt, args.max_prompt_length),
        }

    preprocessed = preprocess_dataset.map(
        preprocess_record,
        num_proc=max(1, args.preprocess_num_proc),
        desc="Normalizing and rendering prompts",
    )
    return list(preprocessed["_normalized_prompt"]), list(preprocessed["_rendered_prompt"])


def shard_prompts(prompts: list[str], shard_count: int) -> list[list[dict[str, Any]]]:
    """Split prompts across rollout instances with simple round-robin balancing."""
    shards = [[] for _ in range(shard_count)]
    for record_index, prompt in enumerate(prompts):
        shard_index = record_index % shard_count
        shards[shard_index].append({"record_index": record_index, "rendered_prompt": prompt})
    return shards


def slice_records(raw_records: list[dict[str, Any]], begin: int, end: int | None) -> tuple[list[dict[str, Any]], int, int]:
    """Slice records using a Python-style half-open interval [begin, end)."""
    total_count = len(raw_records)
    resolved_end = total_count if end is None else min(end, total_count)
    if begin > total_count:
        raise ValueError(f"--begin={begin} is out of range for a dataset with {total_count} rows.")
    if resolved_end < begin:
        raise ValueError(
            f"Resolved slice [{begin}, {resolved_end}) is invalid for a dataset with {total_count} rows."
        )

    sliced_records = raw_records[begin:resolved_end]
    if not sliced_records:
        raise ValueError(f"Slice [{begin}, {resolved_end}) selected no rows from a dataset with {total_count} rows.")
    return sliced_records, begin, resolved_end


def build_worker_command(
    args: argparse.Namespace,
    tensor_parallel_size: int,
    worker_index: int,
    worker_input_path: Path,
    worker_output_path: Path,
    worker_gpu_ids: list[str],
) -> list[str]:
    """Build the coordinator-to-worker command line."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-mode",
        "--model-name-or-path",
        args.model_name_or_path,
        "--worker-index",
        str(worker_index),
        "--worker-input-path",
        str(worker_input_path),
        "--worker-output-path",
        str(worker_output_path),
        "--worker-gpu-ids",
        ",".join(worker_gpu_ids),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--num-samples",
        str(args.num_samples),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--min-p",
        str(args.min_p),
        "--max-prompt-length",
        str(args.max_prompt_length),
        "--max-response-tokens",
        str(args.max_response_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-seqs",
        str(args.max_num_seqs),
    ]
    if args.tokenizer_name_or_path:
        command.extend(["--tokenizer-name-or-path", args.tokenizer_name_or_path])
    command.append("--enforce-eager" if args.enforce_eager else "--no-enforce-eager")
    if args.disable_init_eager_fallback:
        command.append("--disable-init-eager-fallback")
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def build_llm_kwargs(
    args: argparse.Namespace,
    tensor_parallel_size: int,
    *,
    gpu_memory_utilization: float | None = None,
    max_num_seqs: int | None = None,
    enforce_eager: bool | None = None,
) -> dict[str, Any]:
    """Build vLLM constructor kwargs for the worker process."""
    return {
        "model": args.model_name_or_path,
        "tokenizer": args.tokenizer_name_or_path or args.model_name_or_path,
        "trust_remote_code": args.trust_remote_code,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": (
            args.gpu_memory_utilization if gpu_memory_utilization is None else gpu_memory_utilization
        ),
        "max_num_seqs": args.max_num_seqs if max_num_seqs is None else max_num_seqs,
        "max_model_len": args.max_prompt_length + args.max_response_tokens,
        "enforce_eager": args.enforce_eager if enforce_eager is None else enforce_eager,
    }


def build_conservative_llm_kwargs(args: argparse.Namespace, tensor_parallel_size: int) -> dict[str, Any]:
    """Build the safer vLLM startup kwargs used as the fallback path."""
    return build_llm_kwargs(
        args,
        tensor_parallel_size,
        gpu_memory_utilization=min(args.gpu_memory_utilization, _DEFAULT_FALLBACK_GPU_MEMORY_UTILIZATION),
        max_num_seqs=min(args.max_num_seqs, _DEFAULT_FALLBACK_MAX_NUM_SEQS),
        enforce_eager=True,
    )


def resolve_stop_token_ids(tokenizer) -> list[int]:
    """Collect stop token ids so chat-formatted models stop at the assistant boundary."""
    stop_token_ids: list[int] = []

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        stop_token_ids.append(eos_token_id)
    elif isinstance(eos_token_id, (list, tuple)):
        stop_token_ids.extend(token_id for token_id in eos_token_id if isinstance(token_id, int))

    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert_tokens_to_ids):
        im_end_token_id = convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_token_id, int) and im_end_token_id >= 0:
            stop_token_ids.append(im_end_token_id)

    unique_stop_token_ids: list[int] = []
    seen_token_ids: set[int] = set()
    for token_id in stop_token_ids:
        if token_id in seen_token_ids:
            continue
        seen_token_ids.add(token_id)
        unique_stop_token_ids.append(token_id)
    return unique_stop_token_ids


def cleanup_cuda_after_init_failure() -> None:
    """Best-effort CUDA cleanup before retrying vLLM initialization."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def init_llm_with_fallback(args: argparse.Namespace, tensor_parallel_size: int, worker_index: int):
    """Initialize vLLM and retry with conservative settings if startup fails once."""
    from vllm import LLM

    requested_kwargs = build_llm_kwargs(args, tensor_parallel_size)
    fallback_kwargs = build_conservative_llm_kwargs(args, tensor_parallel_size)
    try:
        return LLM(**requested_kwargs)
    except Exception:
        if args.disable_init_eager_fallback or fallback_kwargs == requested_kwargs:
            raise

        print(
            f"Worker {worker_index} failed to initialize vLLM with the requested settings; "
            "retrying with conservative startup settings: "
            f"enforce_eager={fallback_kwargs['enforce_eager']}, "
            f"gpu_memory_utilization={fallback_kwargs['gpu_memory_utilization']}, "
            f"max_num_seqs={fallback_kwargs['max_num_seqs']}."
        )
        cleanup_cuda_after_init_failure()
        try:
            return LLM(**fallback_kwargs)
        except Exception as fallback_error:
            raise RuntimeError(
                f"Worker {worker_index} could not initialize vLLM even after eager fallback."
            ) from fallback_error


def terminate_worker_processes(worker_specs: list[dict[str, Any]]) -> None:
    """Terminate all still-running worker processes."""
    for worker_spec in worker_specs:
        process = worker_spec["process"]
        if process.poll() is None:
            process.terminate()

    for worker_spec in worker_specs:
        process = worker_spec["process"]
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def wait_for_worker_processes(worker_specs: list[dict[str, Any]]) -> None:
    """Wait for all workers and stop the whole launch if any worker fails."""
    pending = worker_specs[:]
    try:
        while pending:
            next_pending = []
            for worker_spec in pending:
                return_code = worker_spec["process"].poll()
                if return_code is None:
                    next_pending.append(worker_spec)
                    continue
                if return_code != 0:
                    raise RuntimeError(
                        f"Worker {worker_spec['worker_index']} failed with exit code {return_code} "
                        f"on GPUs {worker_spec['gpu_ids']}."
                    )
            pending = next_pending
            if pending:
                time.sleep(1.0)
    except Exception:
        terminate_worker_processes(worker_specs)
        raise

    for worker_spec in worker_specs:
        return_code = worker_spec["process"].wait()
        if return_code != 0:
            raise RuntimeError(
                f"Worker {worker_spec['worker_index']} failed with exit code {return_code} "
                f"on GPUs {worker_spec['gpu_ids']}."
            )


def launch_rollout_workers(
    args: argparse.Namespace,
    prompts: list[str],
    launch_layout: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], int]:
    """Launch one worker process per rollout instance and collect their outputs."""
    if not prompts:
        raise ValueError("No prompts are available for rollout generation.")

    max_instance_count = launch_layout["rollout_instance_count"]
    active_instance_count = min(max_instance_count, len(prompts))
    active_gpu_groups = launch_layout["gpu_groups"][:active_instance_count]
    prompt_shards = shard_prompts(prompts, active_instance_count)

    print(
        f"Launching {active_instance_count} rollout worker processes "
        f"(n_gpus_per_node={launch_layout['n_gpus_per_node']}, "
        f"tensor_parallel_size={launch_layout['tensor_parallel_size']})."
    )
    if active_instance_count < max_instance_count:
        print(
            f"Only {active_instance_count} worker processes are needed because the dataset has "
            f"{len(prompts)} prompts."
        )

    worker_specs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="rollouts_vllm_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        for worker_index, (gpu_ids, prompt_shard) in enumerate(zip(active_gpu_groups, prompt_shards)):
            worker_input_path = temp_dir_path / f"worker_{worker_index:02d}_input.json"
            worker_output_path = temp_dir_path / f"worker_{worker_index:02d}_output.json"
            worker_input_path.write_text(json.dumps(prompt_shard, ensure_ascii=False), encoding="utf-8")

            command = build_worker_command(
                args=args,
                tensor_parallel_size=launch_layout["tensor_parallel_size"],
                worker_index=worker_index,
                worker_input_path=worker_input_path,
                worker_output_path=worker_output_path,
                worker_gpu_ids=gpu_ids,
            )
            worker_env = os.environ.copy()
            worker_env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

            print(
                f"Worker {worker_index}: GPUs={gpu_ids}, prompts={len(prompt_shard)}, "
                f"tensor_parallel_size={launch_layout['tensor_parallel_size']}"
            )
            process = subprocess.Popen(command, env=worker_env)
            worker_specs.append(
                {
                    "worker_index": worker_index,
                    "gpu_ids": gpu_ids,
                    "output_path": worker_output_path,
                    "process": process,
                }
            )

        try:
            wait_for_worker_processes(worker_specs)
        except BaseException:
            terminate_worker_processes(worker_specs)
            raise

        rollout_outputs: dict[int, dict[str, Any]] = {}
        for worker_spec in worker_specs:
            worker_payload = json.loads(worker_spec["output_path"].read_text(encoding="utf-8"))
            for item in worker_payload:
                record_index = item["record_index"]
                if record_index in rollout_outputs:
                    raise ValueError(f"Duplicate rollout output for record_index={record_index}")
                rollout_outputs[record_index] = item

    return rollout_outputs, active_instance_count


def run_worker(args: argparse.Namespace) -> None:
    """Worker entry point that owns a single vLLM instance and one GPU group."""
    gpu_ids = parse_gpu_id_list(args.worker_gpu_ids)
    tensor_parallel_size = args.tensor_parallel_size or len(gpu_ids)
    if tensor_parallel_size != len(gpu_ids):
        raise ValueError(
            f"Worker {args.worker_index} received {len(gpu_ids)} GPUs but tensor_parallel_size={tensor_parallel_size}"
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    shard_payload = json.loads(args.worker_input_path.read_text(encoding="utf-8"))
    if not shard_payload:
        args.worker_output_path.write_text("[]", encoding="utf-8")
        return

    from vllm import SamplingParams

    print(f"Worker {args.worker_index} starting on GPUs {gpu_ids} with {len(shard_payload)} prompts.")
    llm = init_llm_with_fallback(args, tensor_parallel_size, args.worker_index)
    stop_token_ids = resolve_stop_token_ids(llm.get_tokenizer())
    sampling_params = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_response_tokens,
        stop_token_ids=stop_token_ids,
    )

    prompts = [item["rendered_prompt"] for item in shard_payload]
    outputs = llm.generate(prompts, sampling_params)
    if len(outputs) != len(shard_payload):
        raise ValueError(
            f"Worker {args.worker_index} produced {len(outputs)} outputs for {len(shard_payload)} prompts."
        )

    worker_results = []
    for item, output in zip(shard_payload, outputs):
        worker_results.append(
            {
                "record_index": item["record_index"],
                "responses": [candidate.text for candidate in output.outputs],
                "finish_reasons": [candidate.finish_reason for candidate in output.outputs],
            }
        )

    args.worker_output_path.parent.mkdir(parents=True, exist_ok=True)
    args.worker_output_path.write_text(json.dumps(worker_results, ensure_ascii=False), encoding="utf-8")
    print(f"Worker {args.worker_index} finished on GPUs {gpu_ids}.")


def run_coordinator(args: argparse.Namespace) -> None:
    """Coordinator entry point that preprocesses prompts and merges worker outputs."""
    raw_records = load_records(args.data, dataset_split=args.dataset_split)
    if not raw_records:
        raise ValueError(f"No prompt records were loaded from {args.data}")
    raw_records, slice_begin, slice_end = slice_records(raw_records, args.begin, args.end)
    print(f"Selected dataset slice [{slice_begin}, {slice_end}) with {len(raw_records)} rows.")

    normalized_prompts, rendered_prompts = preprocess_prompts(args, raw_records)
    launch_layout = resolve_launch_layout(args)
    rollout_outputs, active_instance_count = launch_rollout_workers(args, rendered_prompts, launch_layout)

    if not (len(raw_records) == len(normalized_prompts) == len(rollout_outputs)):
        raise ValueError("Loaded records, normalized prompts, and rollout outputs must align one-to-one.")

    records = []
    for record_index, (source_record, prompt_messages) in enumerate(zip(raw_records, normalized_prompts)):
        output = rollout_outputs.get(record_index)
        if output is None:
            raise ValueError(f"Missing rollout output for record_index={record_index}")

        rollout_record = dict(source_record)
        rollout_record[args.prompt_key] = prompt_messages
        rollout_record["responses"] = output["responses"]
        rollout_record["finish_reasons"] = output["finish_reasons"]
        records.append(rollout_record)

    save_records_as_parquet(records, args.output_path)
    print(f"Saved {len(records)} rollout rows to {args.output_path}")
    print(
        f"Rollout layout summary: GPUs={launch_layout['selected_gpu_ids']}, "
        f"tensor_parallel_size={launch_layout['tensor_parallel_size']}, "
        f"instances={active_instance_count}"
    )
    preview_records(records)


def main() -> None:
    """Dispatch to coordinator mode or worker mode."""
    args = parse_args()
    if args.worker_mode:
        run_worker(args)
    else:
        run_coordinator(args)


if __name__ == "__main__":
    main()
