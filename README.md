# EXL3 CLIP Loader

Built for MOSS-Image with Qwen3.5-9B EXL3. The node outputs a ComfyUI `CLIP` connection for image prompt conditioning. For chat, text generation, or checkpoints such as Gemma, use a dedicated LLM loader.

| Workflow | EXL3 checkpoint | `encoder_type` | Status |
| --- | --- | --- | --- |
| MOSS-Image | Qwen3.5-9B, 4.00 bpw | `auto` | Tested end to end |
| Krea2 | Qwen3-VL-4B | `auto` | Recognized by the loader |
| Fibo2 | Qwen3-VL-4B | `fibo2` | Recognized by the loader |
| Qwen Image 2.1 | Qwen3-VL-8B | `auto` | Recognized by the loader |

Each workflow needs its matching native ComfyUI text encoder.

## Install

1. Install `comfyui-exl3-clip` with Manager. Use a CUDA-capable ComfyUI install. If its Python environment cannot load ExLlamaV3, install the [wheel for your Python, PyTorch, and CUDA build](https://github.com/turboderp-org/exllamav3/releases) in that environment.
2. Download the complete [Qwen3.5-9B EXL3 4.00 bpw revision](https://huggingface.co/turboderp/Qwen3.5-9B-exl3/tree/4.00bpw). Run this from the ComfyUI folder if you use the Hugging Face CLI.

   ```text
   hf download turboderp/Qwen3.5-9B-exl3 --revision 4.00bpw --local-dir models/text_encoders/exl3/Qwen3.5-9B-exl3-4.00bpw
   ```

3. Restart ComfyUI, add **EXL3 CLIP Loader**, and select the downloaded model folder. Connect **CLIP** to your text-encode node. Set the CLIP strength to zero for diffusion-only LoRAs.
