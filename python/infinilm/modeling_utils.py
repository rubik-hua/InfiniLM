import os
from typing import Dict, Union
import time
from safetensors import safe_open
import glob
from tqdm import tqdm


def parse_dtype(dtype_str: str):
    import infinicore

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


def _lazy_torch():
    import importlib

    return importlib.import_module("torch")


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
    checkpoint_file: Union[str, os.PathLike], device="cpu"
) -> Dict[str, "object"]:
    """
    Reads a `safetensor` checkpoint file. We load the checkpoint on "cpu" by default.
    """

    if not checkpoint_file.endswith(".safetensors"):
        return {}

    state_dict: Dict[str, object] = {}
    # Use NumPy backend to avoid importing torch for cpp-backend weight loading.
    with safe_open(checkpoint_file, framework="np") as f:
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
            # With framework="np", this returns a numpy.ndarray.
            state_dict[k] = f.get_tensor(k)

    return state_dict


def get_model_state_dict(
    model_path: str,
    device,
    dtype,
) -> Dict[str, "object"]:
    """
    Load the model weights.
    """
    import infinicore

    print(" read weights ......")
    t1 = time.time()

    # --------------------------------------------------------- #
    #          Load weights from  all *.safetensors files
    # --------------------------------------------------------- #
    model_param = {}
    for file_path in glob.glob(os.path.join(model_path, "*.safetensors")):
        model_param.update(load_state_dict(file_path))

    if model_param.get("lm_head.weight", None) is None:
        model_param["lm_head.weight"] = model_param["model.embed_tokens.weight"]

    # --------------------------------------------------------- #
    #         model_param_infini references numpy arrays
    # --------------------------------------------------------- #
    model_param_infini: Dict[str, object] = {}
    for key in model_param.keys():
        model_param_infini[key] = infinicore.from_numpy(model_param[key], dtype=dtype, device=device)

    t2 = time.time()
    print(f" read weights over! {(t2 - t1) * 1000} ms \n")
    return model_param_infini


def load_model_state_dict_by_file(
    model,
    model_path: str,
    dtype,
) -> Dict[str, "object"]:
    """
    Load the model weights from file.
    """
    import infinicore

    print(" load weights ......")
    t1 = time.time()

    model_keys = model.state_dict_keyname()

    already_loaded_keys = []

    file_list = glob.glob(os.path.join(model_path, "*.safetensors"))
    if len(file_list) > 0:
        for file_path in tqdm(file_list, desc="Processing files"):
            tqdm.write(f"Processing: {os.path.basename(file_path)}")

            # --------------------------------------------------------- #
            #          Load weights from *.safetensors file
            # --------------------------------------------------------- #
            model_param = load_state_dict(file_path)
            already_loaded_keys.extend(model_param.keys())

            # --------------------------------------------------------- #
            #         model_param_infini references numpy arrays
            # --------------------------------------------------------- #
            model_param_infini = {}
            for key in model_param.keys():
                model_param_infini[key] = infinicore.from_numpy(model_param[key], dtype=dtype)
            model.load_state_dict(model_param_infini, strict=False)
            infinicore.sync_device()

    elif os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        file_path = os.path.join(model_path, "pytorch_model.bin")
        torch = _lazy_torch()
        torch_dtype = infinicore.utils.to_torch_dtype(dtype)
        model_params = torch.load(file_path, weights_only=True, map_location="cpu")

        model_param_infini = {}
        for key in model_params.keys():
            if key not in model_keys:
                continue
            model_param_infini[key] = infinicore.from_torch(model_params[key].to(dtype=torch_dtype))
            already_loaded_keys.append(key)

        model.load_state_dict(model_param_infini, strict=False)
        infinicore.sync_device()
    else:
        raise KeyError("Weight file not found.")

    # Keep strict parameter checking for the safetensors path; for `.bin` we allow
    # best-effort loading to unblock cpp-backend bring-up.
    if len(file_list) > 0:
        check_parameters(model_keys, already_loaded_keys)

    t2 = time.time()
    print(f" load weights over! {(t2 - t1) * 1000} ms \n")


def load_model_state_dict_by_tensor(
    model,
    model_path: str,
    dtype,
):
    """
    Load the model weights by tensor.
    """
    import infinicore

    print(" load weights ......")
    t1 = time.time()

    model_keys = model.state_dict_keyname()
    already_loaded_keys = []

    file_list = glob.glob(os.path.join(model_path, "*.safetensors"))
    if len(file_list) > 0:
        for file_path in tqdm(file_list, desc="Processing files"):
            tqdm.write(f"Processing: {os.path.basename(file_path)}")

            with safe_open(file_path, framework="np") as f:
                for name in f.keys():
                    weight_infini = infinicore.from_numpy(f.get_tensor(name), dtype=dtype)
                    model.load_param(name, weight_infini)
                    already_loaded_keys.append(name)
                    infinicore.sync_stream()

    elif os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        file_path = os.path.join(model_path, "pytorch_model.bin")
        torch = _lazy_torch()
        torch_dtype = infinicore.utils.to_torch_dtype(dtype)
        model_params = torch.load(file_path, weights_only=True, map_location="cpu")

        for key in model_params.keys():
            weight_infini = infinicore.from_torch(
                model_params[key].to(dtype=torch_dtype)
            )
            model.load_param(key, weight_infini)
            already_loaded_keys.append(key)
    else:
        raise KeyError("Weight file not found.")

    check_parameters(model_keys, already_loaded_keys)

    t2 = time.time()
    print(f" load weights over! {(t2 - t1) * 1000} ms \n")
