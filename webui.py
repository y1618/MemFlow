#!/usr/bin/env python3
"""
MemFlow Gradio WebUI
Streaming video generation with adaptive memory
"""
import argparse
import os
import tempfile
import gc

# Set PyTorch thread limits before importing torch
# This reduces CPU usage during GPU inference
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import torch
import ffmpeg
import numpy as np
from omegaconf import OmegaConf
from einops import rearrange
import gradio as gr

# Set PyTorch thread limits
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

# Import MemFlow components
from pipeline import CausalInferencePipeline
from utils.misc import set_seed
from utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller

# Global pipeline variable
pipeline = None
device = None
config = None
low_memory = True


def load_model(config_path: str = "configs/inference.yaml"):
    """Load the MemFlow model and pipeline."""
    global pipeline, device, config, low_memory

    try:
        config = OmegaConf.load(config_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {device}")
        if torch.cuda.is_available():
            free_mem = get_cuda_free_memory_gb(device)
            print(f"Free VRAM: {free_mem:.1f} GB")
            low_memory = free_mem < 40

        set_seed(config.seed)
        torch.set_grad_enabled(False)

        # Initialize pipeline
        pipeline = CausalInferencePipeline(config, device=device)

        # Load generator checkpoint
        if config.generator_ckpt and os.path.exists(config.generator_ckpt):
            print(f"Loading generator checkpoint from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict or "generator_ema" in state_dict:
                raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
            elif "model" in state_dict:
                raw_gen_state_dict = state_dict["model"]
            else:
                raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")

            if config.use_ema:
                def _clean_key(name: str) -> str:
                    return name.replace("_fsdp_wrapped_module.", "")
                cleaned_state_dict = {_clean_key(k): v for k, v in raw_gen_state_dict.items()}
                missing, unexpected = pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
                if missing:
                    print(f"[Warning] {len(missing)} parameters missing")
                if unexpected:
                    print(f"[Warning] {len(unexpected)} unexpected parameters")
            else:
                pipeline.generator.load_state_dict(raw_gen_state_dict)

        # LoRA support
        lora_ckpt_path = getattr(config, "lora_ckpt", None)
        if getattr(config, "adapter", None) and lora_ckpt_path and os.path.exists(lora_ckpt_path):
            from utils.lora_utils import configure_lora_for_model
            import peft

            print(f"Applying LoRA from {lora_ckpt_path}")
            pipeline.generator.model = configure_lora_for_model(
                pipeline.generator.model,
                model_name="generator",
                lora_config=config.adapter,
                is_main_process=True,
            )

            lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
            if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
            else:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)
            pipeline.is_lora_enabled = True
            print("LoRA weights loaded")

        # Move to device
        pipeline = pipeline.to(dtype=torch.bfloat16)
        if low_memory:
            DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
        pipeline.generator.to(device=device)
        pipeline.vae.to(device=device)

        print("Model loaded successfully!")
        return "Model loaded successfully!"

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"Error loading model:\n{error_msg}")
        return f"Error loading model: {str(e)}\n\n{error_msg}"


def generate_video(
    prompt: str,
    num_frames: int = 120,
    seed: int = 0,
    progress=gr.Progress()
):
    """Generate a video from the given prompt."""
    global pipeline, device, config, low_memory

    if pipeline is None:
        return None, "Please load the model first!"

    progress(0, desc="Preparing...")
    set_seed(seed)

    # Prepare noise
    sampled_noise = torch.randn(
        [1, num_frames, 16, 60, 104],
        device=device,
        dtype=torch.bfloat16
    )

    prompts = [prompt]

    progress(0.1, desc="Generating video...")

    try:
        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            low_memory=low_memory,
        )

        progress(0.9, desc="Encoding video with NVENC...")

        # Process video
        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        video_output = (255.0 * current_video).clamp(0, 255).to(torch.uint8)

        # Convert to numpy for ffmpeg
        frames_np = video_output[0].numpy()  # Shape: (T, H, W, C)
        height, width = frames_np.shape[1], frames_np.shape[2]

        # Save to temporary file using NVENC hardware encoder
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name

        # Use ffmpeg with NVENC hardware acceleration
        # Convert all frames to bytes at once (more efficient)
        video_bytes = frames_np.tobytes()

        try:
            process = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24',
                       s=f'{width}x{height}', framerate=16)
                .output(output_path, vcodec='h264_nvenc', pix_fmt='yuv420p',
                        preset='p4', cq=23, rc='vbr')
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stderr=True)
            )

            # Write all frames at once
            process.stdin.write(video_bytes)
            process.stdin.close()
            stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
            return_code = process.wait()

            if return_code != 0:
                print(f"ffmpeg stderr:\n{stderr_output}")
                raise RuntimeError(f"ffmpeg encoding failed with return code {return_code}")

        except Exception as e:
            if 'process' in locals():
                try:
                    process.kill()
                except:
                    pass
            print(f"NVENC encoding error: {str(e)}")
            raise e

        # Clean up GPU memory
        # 1. Clear VAE cache
        pipeline.vae.model.clear_cache()

        # 2. Delete large tensors explicitly
        del video, latents, sampled_noise, current_video, video_output

        # 3. Clear KV cache
        pipeline.clear_kv_cache()

        # 4. Force CUDA cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 5. Force Python garbage collection
        gc.collect()

        progress(1.0, desc="Done!")
        return output_path, f"Video generated successfully! Seed: {seed}"

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"Error during generation:\n{error_msg}")

        # Clean up on error as well
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return None, f"Error: {str(e)}\n\n{error_msg}"


def create_ui():
    """Create the Gradio UI."""
    with gr.Blocks(title="MemFlow - Video Generation") as demo:
        gr.Markdown("""
# MemFlow - Streaming Video Generation
Generate consistent long videos with adaptive memory.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Model loading section
                gr.Markdown("### Model")
                config_path = gr.Textbox(
                    label="Config Path",
                    value="configs/inference.yaml",
                    interactive=True
                )
                load_btn = gr.Button("Load Model", variant="primary")
                load_status = gr.Textbox(label="Status", interactive=False)

                gr.Markdown("### Generation Settings")
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your video description...",
                    lines=3
                )

                with gr.Row():
                    num_frames = gr.Slider(
                        minimum=40,
                        maximum=480,
                        value=120,
                        step=40,
                        label="Number of Frames"
                    )
                    seed = gr.Number(
                        value=0,
                        label="Seed",
                        precision=0
                    )

                generate_btn = gr.Button("Generate Video", variant="primary")
                gen_status = gr.Textbox(label="Generation Status", interactive=False)

            with gr.Column(scale=2):
                video_output = gr.Video(label="Generated Video")

        gr.Markdown("""
### Tips
- Maintain consistent subject/background descriptions across prompts for better coherence
- MemFlow supports action changes, object introduction/removal, and background shifts
- Large continuous camera motions can be achieved through cinematic language
        """)

        # Event handlers
        load_btn.click(
            fn=load_model,
            inputs=[config_path],
            outputs=[load_status]
        )

        generate_btn.click(
            fn=generate_video,
            inputs=[prompt, num_frames, seed],
            outputs=[video_output, gen_status]
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MemFlow WebUI")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--config", type=str, default="configs/inference.yaml", help="Config path")
    parser.add_argument("--autoload", action="store_true", help="Auto-load model on startup")
    args = parser.parse_args()

    demo = create_ui()

    if args.autoload:
        print("Auto-loading model...")
        load_model(args.config)

    print(f"Starting WebUI at http://{args.host}:{args.port}")
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft()
    )
