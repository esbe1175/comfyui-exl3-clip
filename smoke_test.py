import argparse
import importlib.util
from pathlib import Path
import sys
import time


parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--model-name", default="Qwen3.5-9B-exl3-4.00bpw")
parser.add_argument("--encoder-type", choices=("auto", "krea2", "fibo2", "moss_image"), default="auto")
args = parser.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
sys.argv = [sys.argv[0]] + (["--cpu"] if args.cpu else [])
root = next(parent for parent in Path(__file__).resolve().parents if (parent / "comfy").is_dir() and (parent / "main.py").is_file())
sys.path.insert(0, str(root))

import comfy.options
comfy.options.enable_args_parsing()

import torch
from comfy.text_encoders.moss_image import MOSSImageTEModel, MOSSImageTokenizer
from comfy.text_encoders.fibo2 import Fibo2TEModel, Fibo2Tokenizer
from comfy.text_encoders.krea2 import Krea2TEModel, Krea2Tokenizer


spec = importlib.util.spec_from_file_location("exl3_clip", Path(__file__).with_name("__init__.py"))
nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes)

encoder_type = args.encoder_type
if encoder_type == "auto":
    encoder_type = "moss_image" if "Qwen3.5" in args.model_name else "krea2"
model_class, tokenizer_class, features = {
    "moss_image": (MOSSImageTEModel, MOSSImageTokenizer, 4096),
    "krea2": (Krea2TEModel, Krea2Tokenizer, 30720),
    "fibo2": (Fibo2TEModel, Fibo2Tokenizer, 46080),
}[encoder_type]
native = model_class(device="meta", dtype=torch.bfloat16)
clip = nodes.ResidentCLIP(native, torch.device("cpu"))
tokenizer = tokenizer_class()
prompts = ["a fluffy fennec fox", "", "  a white cat\n", '{"short_description":"A blue ceramic teapot","text_render":[]}']
for prompt in prompts:
    assert clip.tokenize(prompt) == tokenizer.tokenize_with_weights(prompt)
assert clip.clone().cond_stage_model is native
assert nodes.native_key("lm_head.trellis", "qwen35_9b") is None
assert nodes.native_key("model.language_model.norm.weight", "qwen35_9b") == "qwen35_9b.transformer.model.norm.weight"
print(f"Native {encoder_type} tokenization, checkpoint mapping and CLIP clone passed.", flush=True)
del clip, native

if not args.cpu:
    begin = time.perf_counter()
    clip, = nodes.EXL3CLIPLoader().load_clip(args.model_name, encoder_type=args.encoder_type)
    torch.cuda.synchronize()
    print(f"Loaded {args.model_name} in {time.perf_counter() - begin:.2f}s; {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GiB allocated.", flush=True)
    for prompt in prompts[:3]:
        begin = time.perf_counter()
        tokens = clip.tokenize(prompt)
        output = clip.encode_from_tokens(tokens, return_dict=True)
        cond = output["cond"]
        assert cond.ndim == 3 and cond.shape[0] == 1 and cond.shape[-1] == features, cond.shape
        assert torch.isfinite(cond).all()
        if "attention_mask" in output:
            assert output["attention_mask"].shape == cond.shape[:2]
        if encoder_type == "fibo2":
            assert cond.dtype == torch.bfloat16
        print(f"Prompt {prompt!r}: {tuple(cond.shape)}, {cond.dtype}, {time.perf_counter() - begin:.2f}s; finite.", flush=True)
    image = torch.zeros((1, 64, 64, 3))
    template = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    if encoder_type == "fibo2":
        tokens = clip.tokenize("Make the square blue.", images=[image])
    else:
        tokens = clip.tokenize("a black square", images=[image], llama_template=template, thinking=True)
    encoder_name = "qwen35_9b" if encoder_type == "moss_image" else "qwen3vl_4b"
    assert any(isinstance(t[0], dict) and t[0]["type"] == "image" for group in tokens[encoder_name] for t in group)
    output = clip.encode_from_tokens(tokens, return_dict=True)
    assert torch.isfinite(output["cond"]).all()
    print(f"Image conditioning: {tuple(output['cond'].shape)}; finite.", flush=True)
    assert clip.patcher in nodes.model_management.loaded_models()
    nodes.model_management.unload_model_and_clones(clip.patcher, all_devices=True)
    assert all(t.device == clip.patcher.offload_device for t in clip.cond_stage_model.state_dict().values())
    output = clip.encode_from_tokens(clip.tokenize(prompts[0]), return_dict=True)
    assert torch.isfinite(output["cond"]).all()
    assert clip.patcher in nodes.model_management.loaded_models()
    try:
        clip.patcher.add_patches({encoder_name + ".transformer.model.norm.weight": None})
    except RuntimeError as error:
        assert "strength_clip" in str(error)
    else:
        raise AssertionError("Text-encoder patches were accepted")
    nodes.model_management.unload_model_and_clones(clip.patcher, all_devices=True)
    assert all(t.device == clip.patcher.offload_device for t in clip.cond_stage_model.state_dict().values())
    print("Managed encoder unload/reload passed; unsupported text-encoder weight patches reject clearly.", flush=True)
    del output, cond, clip
