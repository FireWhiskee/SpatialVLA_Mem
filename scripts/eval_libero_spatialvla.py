import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def parse_args():
    parser = argparse.ArgumentParser("Evaluate SpatialVLA / SpatialVLA-Memory on LIBERO.")
    parser.add_argument("--model-path", required=True, help="HF model dir or PEFT adapter checkpoint.")
    parser.add_argument("--base-model-path", default=None, help="Base SpatialVLA checkpoint for PEFT adapters.")
    parser.add_argument("--processor-path", default=None, help="Optional processor/statistics dir. Defaults to model path.")
    parser.add_argument("--suite", default="libero_spatial", choices=sorted(TASK_MAX_STEPS))
    parser.add_argument("--task-ids", default=None, help="Comma-separated task ids. Default: all tasks in suite.")
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--use-memory", action="store_true")
    parser.add_argument("--memory-bank-size", type=int, default=64)
    parser.add_argument("--memory-write-tokens", type=int, default=4)
    parser.add_argument("--memory-retrieve-tokens", type=int, default=8)
    parser.add_argument("--memory-num-heads", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="/root/autodl-tmp/libero_eval_results")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_dtype(name):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def is_peft_checkpoint(path):
    return (Path(path) / "adapter_config.json").exists()


def load_policy(args):
    from model import SpatialVLAConfig, SpatialVLAForConditionalGeneration, SpatialVLAProcessor

    processor_path = args.processor_path or args.model_path
    dtype = torch_dtype(args.dtype)

    if is_peft_checkpoint(args.model_path):
        from peft import PeftModel

        base_path = args.base_model_path
        if base_path is None:
            with open(Path(args.model_path) / "adapter_config.json", "r", encoding="utf-8") as f:
                base_path = json.load(f).get("base_model_name_or_path")
        if not base_path:
            raise ValueError("--base-model-path is required for PEFT checkpoints without base_model_name_or_path.")

        config = SpatialVLAConfig.from_pretrained(base_path, local_files_only=args.local_files_only)
        config.use_memory = bool(args.use_memory)
        config.memory_bank_size = args.memory_bank_size
        config.memory_write_tokens = args.memory_write_tokens
        config.memory_retrieve_tokens = args.memory_retrieve_tokens
        config.memory_num_heads = args.memory_num_heads
        base = SpatialVLAForConditionalGeneration.from_pretrained(
            base_path,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
            local_files_only=args.local_files_only,
        )
        model = PeftModel.from_pretrained(base, args.model_path, local_files_only=args.local_files_only)
        processor = SpatialVLAProcessor.from_pretrained(processor_path, local_files_only=args.local_files_only)
        policy = model.base_model.model
    else:
        config = SpatialVLAConfig.from_pretrained(args.model_path, local_files_only=args.local_files_only)
        if args.use_memory:
            config.use_memory = True
            config.memory_bank_size = args.memory_bank_size
            config.memory_write_tokens = args.memory_write_tokens
            config.memory_retrieve_tokens = args.memory_retrieve_tokens
            config.memory_num_heads = args.memory_num_heads
        model = SpatialVLAForConditionalGeneration.from_pretrained(
            args.model_path,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=args.local_files_only,
        )
        processor = SpatialVLAProcessor.from_pretrained(processor_path, local_files_only=args.local_files_only)
        policy = model

    policy.action_tokenizer = processor.action_tokenizer
    model.to(args.device)
    model.eval()
    return model, policy, processor, dtype


def resolve_unnorm_key(processor, suite, requested):
    candidates = []
    if requested:
        candidates.append(requested)
    candidates.extend([suite, f"{suite}_no_noops", f"{suite}_no_noops/1.0.0", f"{suite}/1.0.0"])
    for key in candidates:
        if key in processor.statistics:
            return key
    raise KeyError(f"No LIBERO action statistics found for {suite}. Available keys: {list(processor.statistics.keys())}")


def libero_image(obs):
    # Match the OpenVLA LIBERO eval convention: rotate the agent-view RGB by 180 degrees.
    return Image.fromarray(obs["agentview_image"][::-1, ::-1])


def dummy_action():
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def normalize_gripper_action(action):
    action = np.asarray(action, dtype=np.float32).copy()
    action[-1] = 1.0 if action[-1] > 0.5 else -1.0
    return action


def invert_gripper_action(action):
    action = np.asarray(action, dtype=np.float32).copy()
    action[-1] = -action[-1]
    return action


@torch.no_grad()
def get_action(policy, processor, image, instruction, unnorm_key, memory_state, dtype):
    inputs = processor(images=image, text=instruction, unnorm_key=unnorm_key, return_tensors="pt")
    inputs = inputs.to(dtype).to(policy.device)
    input_len = inputs["input_ids"].shape[-1]
    if policy.memory_adapter is not None:
        if memory_state is None:
            memory_state = policy.init_memory(inputs["input_ids"].shape[0], device=policy.device, dtype=dtype)
        generated = policy.generate(
            **inputs,
            memory_state=memory_state,
            update_memory=False,
            max_new_tokens=256,
            do_sample=False,
        )
        tokens = generated[:, input_len:]
        image_features = policy.get_image_features(inputs["pixel_values"], inputs.get("intrinsic"))
        _, memory_state = policy.memory_adapter(image_features, memory_state=memory_state, update_memory=True)
    else:
        generated = policy.generate(**inputs, max_new_tokens=256, do_sample=False)
        tokens = generated[:, input_len:]
    decoded = processor.decode_actions(tokens, unnorm_key=unnorm_key)
    action = decoded["actions"][0]
    action = normalize_gripper_action(action)
    action = invert_gripper_action(action)
    return action, memory_state


def run_episode(env, policy, processor, task_description, init_state, args, unnorm_key, dtype):
    env.reset()
    obs = env.set_init_state(init_state)
    memory_state = None
    images = []
    max_steps = args.max_steps or TASK_MAX_STEPS[args.suite]

    for t in range(max_steps + args.num_steps_wait):
        if t < args.num_steps_wait:
            obs, _, done, _ = env.step(dummy_action())
            continue

        image = libero_image(obs)
        if args.save_video:
            images.append(np.asarray(image))
        action, memory_state = get_action(policy, processor, image, task_description, unnorm_key, memory_state, dtype)
        obs, _, done, _ = env.step(action.tolist())
        if done:
            return True, t + 1, images
    return False, max_steps + args.num_steps_wait, images


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    model, policy, processor, dtype = load_policy(args)
    unnorm_key = resolve_unnorm_key(processor, args.suite, args.unnorm_key)
    print(f"Using unnorm_key={unnorm_key}")

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = list(range(suite.n_tasks)) if args.task_ids is None else [int(x) for x in args.task_ids.split(",")]
    log_path = output_dir / f"{args.suite}_{Path(args.model_path).name}_{int(time.time())}.jsonl"

    total_success = 0
    total_episodes = 0
    with log_path.open("w", encoding="utf-8") as log_file:
        for task_id in task_ids:
            task = suite.get_task(task_id)
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env = OffScreenRenderEnv(
                bddl_file_name=bddl,
                camera_heights=args.resolution,
                camera_widths=args.resolution,
            )
            env.seed(args.seed + task_id)
            init_states = suite.get_task_init_states(task_id)
            task_success = 0

            for trial in range(args.num_trials_per_task):
                success, steps, _ = run_episode(
                    env,
                    policy,
                    processor,
                    task.language,
                    init_states[trial % len(init_states)],
                    args,
                    unnorm_key,
                    dtype,
                )
                task_success += int(success)
                total_success += int(success)
                total_episodes += 1
                row = {
                    "suite": args.suite,
                    "task_id": task_id,
                    "task": task.language,
                    "trial": trial,
                    "success": bool(success),
                    "steps": steps,
                    "task_success_rate": task_success / float(trial + 1),
                    "total_success_rate": total_success / float(total_episodes),
                }
                print(json.dumps(row, ensure_ascii=False), flush=True)
                log_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                log_file.flush()

            env.close()

    print(f"Done. episodes={total_episodes} success={total_success} sr={total_success / max(total_episodes, 1):.4f}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
