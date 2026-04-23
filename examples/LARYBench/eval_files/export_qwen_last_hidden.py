import argparse
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.training.trainer_utils.trainer_tools import resize_images


STRIDE_DATASETS = {
    "calvin": "stride_5",
    "vlabench": "stride_5",
    "vlabench_15": "stride_15",
    "vlabench_30": "stride_30",
    "agibotbeta": "stride_45",
    "robocoin": "stride_10",
}

NO_SPLIT_DATASETS = {"human_1st", "robot_1st", "libero"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export StarVLA Qwen last_hidden features to LARYBench-compatible NPZ+CSV")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to StarVLA checkpoint (.pt/.safetensors)")
    parser.add_argument("--input_csv", type=str, required=True, help="Input LARY metadata CSV")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root")
    parser.add_argument("--la_root", type=str, required=True, help="Output latent root, used by LARY_LA_DIR")
    parser.add_argument("--dataset", type=str, default="calvin")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "seen_train", "seen_val", "unseen", "all"],
        default="train",
    )
    parser.add_argument("--model_tag", type=str, default="starvla_qwen_last_hidden")
    parser.add_argument("--stride", type=int, default=5, help="Used for output CSV naming in image-pair datasets")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--instruction", type=str, default="Predict robot action from observations.")
    parser.add_argument("--image_subdir_fallback", type=str, default="images")
    parser.add_argument("--output_csv", type=str, default="", help="Optional explicit output CSV path")
    return parser.parse_args()


def resolve_image_path(path_value: str, data_root: Path, image_subdir_fallback: str) -> Path:
    p = Path(str(path_value))
    if p.is_absolute():
        return p
    direct = data_root / p
    if direct.exists():
        return direct
    return data_root / image_subdir_fallback / p


def resolve_video_path(path_value: str, data_root: Path) -> Path:
    p = Path(str(path_value))
    if p.is_absolute():
        return p
    return data_root / p


def parse_sample_indices(raw: str) -> List[int]:
    text = str(raw).strip().replace("\n", "")
    if not text:
        return []
    parts = [x.strip() for x in text.split(",") if x.strip()]
    return [int(x) for x in parts]


def load_video_frame_pair(video_path: Path, indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV (cv2) is required for video-based datasets such as robot_1st.") from exc

    if not indices:
        raise ValueError(f"Empty sample_indices for video: {video_path}")

    src_idx = int(indices[0])
    tgt_idx = int(indices[-1])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
        ok_src, src_bgr = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, tgt_idx)
        ok_tgt, tgt_bgr = cap.read()
    finally:
        cap.release()

    if not ok_src or src_bgr is None:
        raise RuntimeError(f"Failed to read source frame {src_idx} from {video_path}")
    if not ok_tgt or tgt_bgr is None:
        raise RuntimeError(f"Failed to read target frame {tgt_idx} from {video_path}")

    src_rgb = src_bgr[:, :, ::-1]
    tgt_rgb = tgt_bgr[:, :, ::-1]
    return src_rgb, tgt_rgb


def batched_indices(total: int, batch_size: int) -> Iterable[range]:
    for start in range(0, total, batch_size):
        yield range(start, min(start + batch_size, total))


def pool_hidden(last_hidden: torch.Tensor) -> np.ndarray:
    pooled = last_hidden.mean(dim=1)
    return pooled.detach().cpu().numpy()


@torch.inference_mode()
def encode_single_image_features(model, examples: List[dict]) -> np.ndarray:
    batch_images = [to_pil_preserve(example["image"]) for example in examples]
    instructions = [example["lang"] for example in examples]

    train_obs_image_size = getattr(model.config.framework, "obs_image_size", None)
    if train_obs_image_size:
        batch_images = resize_images(batch_images, target_size=train_obs_image_size)

    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.qwen_vl_interface(
            **qwen_inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
    return pool_hidden(outputs.hidden_states[-1])


def default_output_csv(args: argparse.Namespace) -> Path:
    artifacts_dir = Path("examples/LARYBench/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset in STRIDE_DATASETS:
        name = f"{args.split}_la_{args.dataset}_{args.stride}_{args.model_tag}.csv"
    else:
        name = f"{args.split}_la_{args.dataset}_{args.model_tag}.csv"
    return artifacts_dir / name


def build_save_dir(la_root: Path, dataset: str, split: str, model_tag: str, stride: int) -> Path:
    dataset = dataset.lower()
    if dataset in NO_SPLIT_DATASETS:
        return la_root / dataset / model_tag

    if dataset in STRIDE_DATASETS:
        stride_dir = STRIDE_DATASETS[dataset]
        return la_root / dataset / stride_dir / split / model_tag

    return la_root / dataset / split / model_tag


def infer_mode(df: pd.DataFrame) -> str:
    if {"src_img", "tgt_img"}.issubset(df.columns):
        return "image_pair"
    if {"video_path", "sample_indices"}.issubset(df.columns):
        return "video_pair"
    raise ValueError(
        "Unsupported CSV schema. Need either (src_img,tgt_img) or (video_path,sample_indices). "
        f"Columns got: {list(df.columns)}"
    )


def make_la_csv_path(save_path: Path) -> str:
    # Use absolute path to avoid resolver/model-name mismatch in downstream LARY loaders.
    return str(save_path.resolve())


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    la_root = Path(args.la_root)
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv) if args.output_csv else default_output_csv(args)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    if "la_path" not in df.columns:
        df["la_path"] = ""

    mode = infer_mode(df)

    model = baseframework.from_pretrained(args.ckpt_path)
    if not hasattr(model, "qwen_vl_interface"):
        raise RuntimeError("Loaded framework has no qwen_vl_interface; this exporter currently supports Qwen-based frameworks.")
    model = model.to("cuda").eval()

    save_dir = build_save_dir(la_root, args.dataset, args.split, args.model_tag, args.stride)
    save_dir.mkdir(parents=True, exist_ok=True)

    total_batches = (len(df) + args.batch_size - 1) // args.batch_size
    for batch_range in tqdm(batched_indices(len(df), args.batch_size), total=total_batches, desc=f"Export {args.split}"):
        src_examples: List[dict] = []
        tgt_examples: List[dict] = []
        row_indices: List[int] = []

        for idx in batch_range:
            row = df.iloc[idx]

            if mode == "image_pair":
                src_path = resolve_image_path(row["src_img"], data_root, args.image_subdir_fallback)
                tgt_path = resolve_image_path(row["tgt_img"], data_root, args.image_subdir_fallback)
                src_img = np.asarray(Image.open(src_path).convert("RGB"))
                tgt_img = np.asarray(Image.open(tgt_path).convert("RGB"))
            else:
                video_path = resolve_video_path(row["video_path"], data_root)
                frame_indices = parse_sample_indices(row["sample_indices"])
                src_img, tgt_img = load_video_frame_pair(video_path, frame_indices)

            src_examples.append({"image": [src_img], "lang": args.instruction})
            tgt_examples.append({"image": [tgt_img], "lang": args.instruction})
            row_indices.append(idx)

        src_feat = encode_single_image_features(model, src_examples)
        tgt_feat = encode_single_image_features(model, tgt_examples)
        latent = tgt_feat - src_feat

        for i, row_idx in enumerate(row_indices):
            save_name = f"latent_action_{row_idx:08d}.npz"
            save_path = save_dir / save_name
            tokens = latent[i][None, :]
            np.savez_compressed(save_path, tokens=tokens, indices=np.empty((0,), dtype=np.int64))

            la_value = make_la_csv_path(save_path)
            df.at[row_idx, "la_path"] = la_value

    df.to_csv(output_csv, index=False)
    print(f"Saved CSV: {output_csv}")
    print(f"Saved NPZ root: {save_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
