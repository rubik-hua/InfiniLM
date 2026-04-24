import os
import json
import math
from typing import Dict, Union
import time
import torch
from safetensors import safe_open
import glob
from tqdm import tqdm
import infinicore


def parse_dtype(dtype_str: str):
    if dtype_str == "float32":
        return infinicore.float32
    elif dtype_str == "float16":
        return infinicore.float16
    elif dtype_str == "bfloat16":
        return infinicore.bfloat16
    elif dtype_str == "int8":
        return infinicore.int8
    elif dtype_str == "int32":
        return infinicore.int32
    elif dtype_str == "int64":
        return infinicore.int64
    else:
        raise ValueError(f"Unknown dtype string: {dtype_str}")


str_to_torch_dtype = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I32": torch.int32,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


def check_parameters(model_keys: list, already_loaded_keys: list):
    model_keys = set(model_keys)
    already_loaded_keys = set(already_loaded_keys)
    intersection = model_keys & already_loaded_keys

    missing_keys = model_keys - intersection
    unexpected_keys = already_loaded_keys - intersection
    error_msgs: list[str] = []

    if len(unexpected_keys) > 0:
        error_msgs.append(
            "Unexpected key(s) in state_dict: {}. ".format(
                ", ".join('"{}"'.format(k) for k in unexpected_keys)
            )
        )
    if len(missing_keys) > 0:
        error_msgs.append(
            "Missing key(s) in state_dict: {}. ".format(
                ", ".join('"{}"'.format(k) for k in missing_keys)
            )
        )

    if len(error_msgs) > 0:
        raise RuntimeError(
            "Error(s) in loading state_dict\n\t{}".format("\n\t".join(error_msgs))
        )


def load_state_dict(
    checkpoint_file: Union[str, os.PathLike], device="cpu", dtype=torch.bfloat16
) -> Dict[str, torch.Tensor]:
    """
    Reads a `safetensor` checkpoint file. We load the checkpoint on "cpu" by default.
    """

    if not checkpoint_file.endswith(".safetensors"):
        return {}

    state_dict = {}
    with safe_open(checkpoint_file, framework="pt") as f:
        metadata = f.metadata()
        if metadata is not None and metadata.get("format") not in [
            "pt",
            "tf",
            "flax",
            "mlx",
        ]:
            raise OSError(
                f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata."
            )

        for k in f.keys():
            state_dict[k] = f.get_tensor(k).to(device=device)

    return state_dict


def get_model_state_dict(
    model_path: str,
    device: infinicore.device,
    dtype=infinicore.dtype,
) -> Dict[str, infinicore.Tensor]:
    """
    Load the model weights.
    """

    print(" read weights ......")
    t1 = time.time()

    torch_device = device.type
    torch_dtype = infinicore.utils.to_torch_dtype(dtype)

    # --------------------------------------------------------- #
    #          Load weights from  all *.safetensors files
    # --------------------------------------------------------- #
    model_param = {}
    for file_path in glob.glob(os.path.join(model_path, "*.safetensors")):
        model_param.update(
            load_state_dict(file_path, device=torch_device, dtype=torch_dtype)
        )

    if model_param.get("lm_head.weight", None) is None:
        model_param["lm_head.weight"] = model_param["model.embed_tokens.weight"]

    # --------------------------------------------------------- #
    #         model_param_infini references torch.Tensor
    # --------------------------------------------------------- #
    model_param_infini = {}
    for key in model_param.keys():
        model_param_infini[key] = infinicore.from_torch(model_param[key])

    t2 = time.time()
    print(f" read weights over! {(t2 - t1) * 1000} ms \n")
    return model_param_infini


def _get_mup_scales(model_path: str) -> Dict[str, float]:
    """Detect MuP scaling factors from config.json for fm9g models.

    Returns a dict with keys: scale_input, scale_output, scale_down, scale_lm_head.
    All values default to 1.0 (no-op) if the model doesn't use MuP scaling.
    """
    scales = {"scale_input": 1.0, "scale_output": 1.0, "scale_down": 1.0, "scale_lm_head": 1.0}
    try:
        with open(os.path.join(model_path, "config.json")) as f:
            cfg = json.load(f)
        model_type = cfg.get("model_type", "")
        if model_type != "fm9g":
            return scales
        if "scale_emb" not in cfg or "scale_depth" not in cfg:
            return scales
        scales["scale_input"] = float(cfg["scale_emb"])
        proj_scale = float(cfg["scale_depth"]) / math.sqrt(float(cfg["num_hidden_layers"]))
        scales["scale_output"] = proj_scale
        scales["scale_down"] = proj_scale
        if "dim_model_base" in cfg and "hidden_size" in cfg:
            scales["scale_lm_head"] = float(cfg["dim_model_base"]) / float(cfg["hidden_size"])
    except Exception as e:
        import logging
        logging.warning(f"Failed to detect MuP scales from config: {e}")
    return scales


def _apply_mup_scales(model_param: Dict[str, torch.Tensor], scales: Dict[str, float]) -> None:
    """Apply MuP scaling factors to model weights in-place."""
    if scales["scale_input"] != 1.0 and "model.embed_tokens.weight" in model_param:
        model_param["model.embed_tokens.weight"] = model_param["model.embed_tokens.weight"] * scales["scale_input"]
    if scales["scale_output"] != 1.0 or scales["scale_down"] != 1.0:
        for k, v in list(model_param.items()):
            if scales["scale_output"] != 1.0 and k.endswith(".self_attn.o_proj.weight"):
                model_param[k] = v * scales["scale_output"]
            elif scales["scale_down"] != 1.0 and k.endswith(".mlp.down_proj.weight"):
                model_param[k] = v * scales["scale_down"]
    if scales["scale_lm_head"] != 1.0 and "lm_head.weight" in model_param:
        model_param["lm_head.weight"] = model_param["lm_head.weight"] * scales["scale_lm_head"]


def load_model_state_dict_by_file(
    model: infinicore.nn.Module,
    model_path: str,
    dtype=infinicore.dtype,
) -> Dict[str, infinicore.Tensor]:
    """
    Load the model weights from file.
    """
    print(" load weights ......")
    t1 = time.time()

    torch_device = "cpu"
    torch_dtype = infinicore.utils.to_torch_dtype(dtype)
    model_keys = model.state_dict_keyname()

    scales = _get_mup_scales(model_path)

    already_loaded_keys = []
    embed_tokens_tensor = None

    file_list = glob.glob(os.path.join(model_path, "*.safetensors"))
    if len(file_list) > 0:
        for file_path in tqdm(file_list, desc="Processing files"):
            tqdm.write(f"Processing: {os.path.basename(file_path)}")

            # --------------------------------------------------------- #
            #          Load weights from *.safetensors file
            # --------------------------------------------------------- #
            model_param = load_state_dict(
                file_path, device=torch_device, dtype=torch_dtype
            )
            already_loaded_keys.extend(model_param.keys())

            _apply_mup_scales(model_param, scales)

            # --------------------------------------------------------- #
            #         model_param_infini references torch.Tensor
            # --------------------------------------------------------- #
            model_param_infini = {}
            for key in model_param.keys():
                model_param_infini[key] = infinicore.from_torch(model_param[key])
            if "model.embed_tokens.weight" in model_param_infini:
                embed_tokens_tensor = model_param_infini["model.embed_tokens.weight"]
            model.load_state_dict(model_param_infini, strict=False)
            infinicore.sync_device()

    elif os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        file_path = os.path.join(model_path, "pytorch_model.bin")
        model_params = torch.load(file_path, weights_only=True, map_location="cpu")

        _apply_mup_scales(model_params, scales)

        model_param_infini = {}
        for key in model_params.keys():
            model_param_infini[key] = infinicore.from_torch(
                model_params[key].to(dtype=torch_dtype)
            )
            already_loaded_keys.append(key)

        if "model.embed_tokens.weight" in model_param_infini:
            embed_tokens_tensor = model_param_infini["model.embed_tokens.weight"]
        model.load_state_dict(model_param_infini, strict=True)
        infinicore.sync_device()
    else:
        raise KeyError("Weight file not found.")

    # Handle tied weights: if lm_head.weight is missing, share embed_tokens.weight
    if "lm_head.weight" in model_keys and "lm_head.weight" not in already_loaded_keys:
        if embed_tokens_tensor is not None:
            model.load_state_dict({"lm_head.weight": embed_tokens_tensor}, strict=False)
            already_loaded_keys.append("lm_head.weight")

    check_parameters(model_keys, already_loaded_keys)

    t2 = time.time()
    print(f" load weights over! {(t2 - t1) * 1000} ms \n")


def load_model_state_dict_by_tensor(
    model: infinicore.nn.Module,
    model_path: str,
    dtype=infinicore.dtype,
):
    """
    Load the model weights by tensor.
    """

    print(" load weights ......")
    t1 = time.time()

    torch_dtype = infinicore.utils.to_torch_dtype(dtype)
    model_keys = model.state_dict_keyname()
    already_loaded_keys = []
    embed_tokens_tensor = None

    scales = _get_mup_scales(model_path)

    file_list = glob.glob(os.path.join(model_path, "*.safetensors"))
    if len(file_list) > 0:
        for file_path in tqdm(file_list, desc="Processing files"):
            tqdm.write(f"Processing: {os.path.basename(file_path)}")

            with safe_open(file_path, "pt", "cpu") as f:
                for name in f.keys():
                    tensor = f.get_tensor(name).to(dtype=torch_dtype)

                    # Apply MuP scaling per-tensor
                    if scales["scale_input"] != 1.0 and name == "model.embed_tokens.weight":
                        tensor = tensor * scales["scale_input"]
                    elif scales["scale_lm_head"] != 1.0 and name == "lm_head.weight":
                        tensor = tensor * scales["scale_lm_head"]
                    elif scales["scale_output"] != 1.0 and name.endswith(".self_attn.o_proj.weight"):
                        tensor = tensor * scales["scale_output"]
                    elif scales["scale_down"] != 1.0 and name.endswith(".mlp.down_proj.weight"):
                        tensor = tensor * scales["scale_down"]

                    weight_infini = infinicore.from_torch(tensor)
                    if name == "model.embed_tokens.weight":
                        embed_tokens_tensor = weight_infini
                    model.load_param(name, weight_infini)
                    already_loaded_keys.append(name)
                    infinicore.sync_stream()

    elif os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        file_path = os.path.join(model_path, "pytorch_model.bin")
        model_params = torch.load(file_path, weights_only=True, map_location="cpu")

        _apply_mup_scales(model_params, scales)

        for key in model_params.keys():
            weight_infini = infinicore.from_torch(
                model_params[key].to(dtype=torch_dtype)
            )
            if key == "model.embed_tokens.weight":
                embed_tokens_tensor = weight_infini
            model.load_param(key, weight_infini)
            already_loaded_keys.append(key)
    else:
        raise KeyError("Weight file not found.")

    # Handle tied weights: if lm_head.weight is missing, share embed_tokens.weight
    if "lm_head.weight" in model_keys and "lm_head.weight" not in already_loaded_keys:
        if embed_tokens_tensor is not None:
            model.load_param("lm_head.weight", embed_tokens_tensor)
            already_loaded_keys.append("lm_head.weight")

    check_parameters(model_keys, already_loaded_keys)

    t2 = time.time()
    print(f" load weights over! {(t2 - t1) * 1000} ms \n")
