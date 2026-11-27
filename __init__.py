import copy
from importlib import import_module
import json
import logging
import os

import torch
from safetensors import safe_open

import comfy.hooks
import comfy.model_management as model_management
import comfy.model_patcher
import comfy.ops
import comfy.sd
import folder_paths


MISSING_MODEL = "(install an EXL3 text encoder)"


class ResidentOperations(comfy.ops.manual_cast):
    class Linear(comfy.ops.manual_cast.Linear):
        # Direct checkpoint loading needs parameter slots even with Aimdo enabled.
        _load_from_state_dict = torch.nn.Module._load_from_state_dict


class EXL3Linear(comfy.ops.manual_cast.Linear):
    comfy_force_forward = True

    def __init__(self, in_features, out_features, tensors):
        torch.nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        prefix = next(iter(tensors)).rsplit(".", 1)[0]
        for name in ("trellis", "suh", "svh", "su", "sv", "bias", "mcg", "mul1"):
            metadata = tensors.get(prefix + "." + name)
            parameter = None
            if metadata is not None:
                parameter = torch.nn.Parameter(torch.empty(metadata["shape"], dtype=metadata["torch_dtype"], device="meta"), requires_grad=False)
            self.register_parameter("weight" if name == "trellis" else name, parameter)

    def forward(self, input):
        from exllamav3.modules.quant.exl3 import LinearEXL3

        with comfy.ops.CastBiasWeightContext(self, input, dtype=torch.int16, bias_dtype=torch.float16, offloadable=True) as (packed, bias):
            tensors = {name: model_management.cast_to(parameter, None, input.device)
                       for name in ("suh", "svh", "su", "sv", "mcg", "mul1")
                       if (parameter := getattr(self, name)) is not None}
            linear = LinearEXL3(None, self.in_features, self.out_features, trellis=packed,
                                bias=bias, out_dtype=torch.float16, transformers_fix=True, **tensors)
            return linear.forward(input.to(torch.float16).contiguous(), {}).to(input.dtype)


def model_directories():
    models = {}
    for root in folder_paths.get_folder_paths("text_encoders"):
        for parent in (os.path.join(root, "exl3"), root):
            if not os.path.isdir(parent):
                continue
            for name in sorted(os.listdir(parent)):
                directory = os.path.join(parent, name)
                config = os.path.join(directory, "config.json")
                if name in models or not os.path.isfile(config):
                    continue
                if not folder_paths.is_within_directory(parent, config):
                    continue
                try:
                    with open(config, encoding="utf-8") as file:
                        metadata = json.load(file)
                    if metadata.get("quantization_config", {}).get("quant_method") == "exl3":
                        models[name] = os.path.realpath(directory)
                except (OSError, ValueError, AttributeError):
                    continue
    return models


def model_names():
    return sorted(model_directories()) or [MISSING_MODEL]


def model_directory(model_name):
    path = model_directories().get(model_name)
    if path is None:
        raise FileNotFoundError(
            "Download the complete EXL3 revision into models/text_encoders/exl3/<model-name>, "
            "including config.json and the safetensors shards, then refresh the node."
        )
    return path


def native_key(key, encoder_name):
    if key == "lm_head" or key.startswith("lm_head."):
        return None
    if key.endswith(".trellis"):
        key = key[:-len(".trellis")] + ".weight"
    if key.startswith("model.language_model."):
        return encoder_name + ".transformer.model." + key[len("model.language_model."):]
    if key.startswith("model.visual."):
        return encoder_name + ".transformer.visual." + key[len("model.visual."):]
    raise ValueError(f"Unsupported EXL3 checkpoint tensor: {key}")


def encoder_target(config, encoder_type="auto"):
    if encoder_type not in ("auto", "krea2", "fibo2", "moss_image", "qwen_image"):
        raise ValueError(f"Unknown EXL3 conditioning type: {encoder_type!r}.")
    if config.get("quantization_config", {}).get("quant_method") != "exl3":
        raise ValueError("EXL3 CLIP requires an EXL3-quantized checkpoint.")
    text_config = config.get("text_config", {})
    model_type = config.get("model_type")
    hidden_size = text_config.get("hidden_size")
    model_options = {"custom_operations": ResidentOperations}
    fields = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "head_dim", "vocab_size", "rms_norm_eps")
    if model_type == "qwen3_vl" and hidden_size in (2560, 4096):
        from comfy.text_encoders.llama import Qwen3VL_4BConfig, Qwen3VL_8BConfig
        from comfy.text_encoders.qwen3vl import QWEN3VL_VISION, QWEN3VL_VISION_COMMON

        if hidden_size == 4096:
            from comfy.text_encoders.qwen_image21 import QwenImage21TEModel

            encoder_name = "qwen3vl_8b"
            native_config = Qwen3VL_8BConfig()
            supported_types = ("qwen_image",)
            encoder_class = QwenImage21TEModel
            model_options["qwen3vl_8b_model_config"] = {"lm_head": False}
        else:
            from comfy.text_encoders.krea2 import Krea2TEModel

            encoder_name = "qwen3vl_4b"
            native_config = Qwen3VL_4BConfig()
            supported_types = ("krea2", "fibo2")
            if encoder_type == "fibo2":
                from comfy.text_encoders.fibo2 import Fibo2TEModel

                encoder_class = Fibo2TEModel
            else:
                encoder_class = Krea2TEModel
        vision_config = {**QWEN3VL_VISION_COMMON, **QWEN3VL_VISION[encoder_name], "out_hidden_size": hidden_size}
        rope_key = "rope_scaling"
        expected_rope = {"mrope_interleaved": native_config.interleaved_mrope, "mrope_section": native_config.rope_dims, "rope_type": "default"}
        fields += ("rope_theta",)
    elif model_type == "qwen3_5" and hidden_size == 4096:
        from comfy.text_encoders.moss_image import MOSSImageTEModel
        from comfy.text_encoders.qwen35 import _make_config, QWEN35_MODELS, QWEN35_VISION_DEFAULTS

        encoder_name = "qwen35_9b"
        native_config = _make_config("qwen35_9b")
        supported_types = ("moss_image",)
        encoder_class = MOSSImageTEModel
        model_options["qwen35_9b_model_config"] = {"lm_head": False}
        vision_config = {**QWEN35_VISION_DEFAULTS, **QWEN35_MODELS[encoder_name]["vision"], "out_hidden_size": hidden_size}
        rope_key = "rope_parameters"
        expected_rope = {k: getattr(native_config, k) for k in ("rope_theta", "partial_rotary_factor", "mrope_section")}
        expected_rope.update(mrope_interleaved=True, rope_type="default")
        fields += ("linear_key_head_dim", "linear_num_key_heads", "linear_num_value_heads", "linear_value_head_dim", "layer_types")
    else:
        raise ValueError(f"Unsupported EXL3 encoder architecture: model_type={model_type!r}, hidden_size={hidden_size!r}, layers={text_config.get('num_hidden_layers')!r}. Supported: Qwen3-VL-4B, Qwen3-VL-8B, Qwen3.5-9B.")
    if encoder_type == "auto":
        encoder_type = supported_types[0]
    if encoder_type not in supported_types:
        raise ValueError(f"EXL3 {encoder_name} cannot use conditioning type {encoder_type!r}. Use auto or {', '.join(supported_types)}.")

    expected_text = {k: getattr(native_config, k) for k in fields}
    expected_text.update(hidden_act=native_config.mlp_activation, attention_bias=native_config.qkv_bias)
    if model_type == "qwen3_5":
        expected_text["linear_conv_kernel_dim"] = native_config.conv_kernel_size
    checks = (("text_config", text_config, expected_text),
              (f"text_config.{rope_key}", text_config.get(rope_key, {}), expected_rope),
              ("vision_config", config.get("vision_config", {}), {**vision_config, "hidden_act": "gelu_pytorch_tanh"}))
    mismatches = [f"{section}.{key}: expected {value!r}, got {actual.get(key)!r}"
                  for section, actual, expected in checks for key, value in expected.items() if actual.get(key) != value]
    if mismatches:
        raise ValueError(f"Unsupported EXL3 {encoder_name} configuration: {'; '.join(mismatches)}")
    logging.info("EXL3 detected %s; conditioning type: %s.", encoder_name, encoder_type)
    return encoder_class, model_options


def load_encoder(directory, device, encoder_type="auto"):
    try:
        from exllamav3.loader import SafetensorsCollection
    except ImportError as error:
        raise RuntimeError(
            "Install the ExLlamaV3 wheel that matches this ComfyUI Python, PyTorch and CUDA build, "
            "then restart ComfyUI. See the EXL3 CLIP Loader README."
        ) from error

    with open(os.path.join(directory, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    encoder_class, model_options = encoder_target(config, encoder_type)

    filenames = [filename for filename in os.listdir(directory) if filename.endswith(".safetensors")]
    if not filenames:
        raise FileNotFoundError("The EXL3 folder contains config.json but no safetensors shards.")
    index = os.path.join(directory, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index, encoding="utf-8") as file:
            referenced = set(json.load(file)["weight_map"].values())
        if missing := referenced.difference(filenames):
            raise FileNotFoundError(f"Download the missing EXL3 shards: {', '.join(sorted(missing))}")
    for filename in filenames:
        if not folder_paths.is_within_directory(directory, os.path.join(directory, filename)):
            raise ValueError("EXL3 checkpoint file escapes its model directory.")
    collection = SafetensorsCollection(directory)
    model = encoder_class(device="meta", dtype=torch.bfloat16, model_options=model_options)
    encoder_name = model.clip_name
    linears = []
    for key in collection.tensor_file_map:
        if key.endswith(".trellis"):
            name = native_key(key[:-len(".trellis")], encoder_name)
            if name is None:
                continue
            module = model.get_submodule(name)
            if not isinstance(module, torch.nn.Linear):
                raise ValueError(f"EXL3 projection is not a native linear layer: {key}")
            replacement = EXL3Linear(module.in_features, module.out_features, collection.list_tensors(key[:-len(".trellis")]))
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent), child, replacement)
            linears.append(replacement)

    expected = model.state_dict()
    expected.pop(encoder_name + ".logit_scale")
    sources = {}
    memory = 0
    for key, filename in collection.tensor_file_map.items():
        name = native_key(key, encoder_name)
        if name is None:
            continue
        if name not in expected:
            raise ValueError(f"Unexpected EXL3 checkpoint tensor: {key}")
        metadata = collection.get_tensor_meta(key)[key]
        if tuple(metadata["shape"]) != tuple(expected[name].shape):
            raise ValueError(f"EXL3 checkpoint shape mismatch: {key}")
        memory += metadata["n_bytes"]
        sources.setdefault(filename, []).append((key, name))
    missing = set(expected) - {name for entries in sources.values() for _, name in entries}
    if missing:
        raise ValueError(f"EXL3 checkpoint is missing encoder tensors: {', '.join(sorted(missing)[:5])}")
    collection.close()

    initial_device = model_management.text_encoder_initial_device(device, model_management.text_encoder_offload_device(), memory)
    for filename, entries in sources.items():
        with safe_open(filename, framework="pt", device=str(initial_device)) as f:
            for key, name in entries:
                parent, _, child = name.rpartition(".")
                module = model.get_submodule(parent)
                tensor = f.get_tensor(key)
                if child in module._parameters:
                    tensor = torch.nn.Parameter(tensor, requires_grad=module._parameters[child].requires_grad)
                setattr(module, child, tensor)
    clip_model = model.get_submodule(encoder_name)
    logit_scale = clip_model.logit_scale
    clip_model.logit_scale = torch.nn.Parameter(logit_scale.to(initial_device), requires_grad=logit_scale.requires_grad)
    model.device = initial_device
    model_management.archive_model_dtypes(model)
    logging.info("EXL3 %s text encoder: %d quantized projections, %.2f GiB of weights.", encoder_name, len(linears), memory / 1024 ** 3)
    return model


class ResidentModelPatcher(comfy.model_patcher.ModelPatcher):
    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        keys = [key if isinstance(key, str) else key[0] for key in patches]
        if strength_patch != 0 and any(key.startswith(("qwen3vl_4b.", "qwen3vl_8b.", "qwen35_9b.")) for key in keys):
            raise RuntimeError("EXL3 CLIP does not support text-encoder LoRAs. Set strength_clip to 0.")
        return []

    def add_hook_patches(self, hook, patches, strength_patch=1.0, strength_model=1.0):
        return self.add_patches(patches, strength_patch, strength_model)

    def patch_hooks(self, hooks):
        if hooks is not None and self.hook_patches:
            raise RuntimeError("EXL3 CLIP does not support text-encoder weight hooks.")

    def model_state_dict_for_saving(self, model=None, prefix=""):
        raise RuntimeError("EXL3 CLIP export is unsupported; use the original EXL3 checkpoint directory.")


class ResidentCLIP(comfy.sd.CLIP):
    def __init__(self, model, device):
        super().__init__(no_init=True)
        self.cond_stage_model = model
        encoder_class = type(model)
        module = import_module(encoder_class.__module__)
        tokenizer_class = getattr(module, encoder_class.__name__.removesuffix("TEModel") + "Tokenizer")
        self.tokenizer = tokenizer_class(embedding_directory=folder_paths.get_folder_paths("embeddings"))
        self.patcher = ResidentModelPatcher(model, load_device=device, offload_device=model_management.text_encoder_offload_device())
        self.patcher.is_clip = True
        self.patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
        self.layer_idx = None
        self.use_clip_schedule = False
        self.apply_hooks_to_conds = None
        self.tokenizer_options = {}

    def clone(self, disable_dynamic=False):
        clone = copy.copy(self)
        clone.patcher = self.patcher.clone(disable_dynamic=disable_dynamic)
        clone.tokenizer_options = self.tokenizer_options.copy()
        return clone

    def load_sd(self, sd, full_model=False):
        raise RuntimeError("EXL3 CLIP must be loaded through EXL3 CLIP Loader.")

    def generate(self, tokens, do_sample=True, max_length=256, temperature=1.0, top_k=50, top_p=0.95, min_p=0.0, repetition_penalty=1.0, seed=None, presence_penalty=0.0, mtp=True):
        raise RuntimeError("EXL3 CLIP text generation is unsupported; use a dedicated LLM loader.")


class EXL3CLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_name": (model_names(),)},
                "optional": {"encoder_type": (["auto", "krea2", "fibo2", "moss_image", "qwen_image"], {"default": "auto", "advanced": True, "tooltip": "Auto detects the encoder architecture and selects its default conditioning. Override only when the same checkpoint supports another conditioning format."})}}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "model/loaders"
    DESCRIPTION = "Load a supported EXL3 text encoder as CLIP. Detects the architecture from config.json and uses ComfyUI memory management for loading and offloading."

    def load_clip(self, model_name, encoder_type="auto"):
        directory = model_directory(model_name)
        device = model_management.get_torch_device()
        if device.type != "cuda":
            raise RuntimeError("EXL3 CLIP requires a CUDA GPU.")
        logging.info("Loading EXL3 text encoder %s.", model_name)
        model = load_encoder(directory, device, encoder_type)
        return (ResidentCLIP(model, device),)


NODE_CLASS_MAPPINGS = {"EXL3CLIPLoader": EXL3CLIPLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"EXL3CLIPLoader": "EXL3 CLIP Loader"}
