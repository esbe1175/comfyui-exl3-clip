# EXL3 CLIP Loader

Loads EXL3 text encoders into ComfyUI's native text-conditioning path. For MOSS-Image, use Qwen3.5-9B EXL3 and leave **encoder_type** on **auto**.

## Install

1. Install `comfyui-exl3-clip` with Manager. Use a CUDA-capable ComfyUI install. If its Python environment cannot load ExLlamaV3, install the [wheel for your Python, PyTorch, and CUDA build](https://github.com/turboderp-org/exllamav3/releases) in that environment.
2. Download the complete [Qwen3.5-9B EXL3 4.00 bpw revision](https://huggingface.co/turboderp/Qwen3.5-9B-exl3/tree/4.00bpw). Run this from the ComfyUI folder if you use the Hugging Face CLI.

   ```text
   hf download turboderp/Qwen3.5-9B-exl3 --revision 4.00bpw --local-dir models/text_encoders/exl3/Qwen3.5-9B-exl3-4.00bpw
   ```

3. Restart ComfyUI, add **EXL3 CLIP Loader**, and select the downloaded model folder. Connect **CLIP** to your text-encode node. Set the CLIP strength to zero for diffusion-only LoRAs.

The loader also recognizes Qwen3-VL-4B for Krea2 and Fibo2, and Qwen3-VL-8B for Qwen Image 2.1 when the matching native encoder is available. Select **fibo2** explicitly for Fibo2.
