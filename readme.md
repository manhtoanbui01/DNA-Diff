# DNA-Diff

Diffusion-based synthetic data generation for **night-time person re-identification**.

DNA-Diff fine-tunes Stable Diffusion v1.5 in three stages so that it can render a
given identity, taken from a daytime reference image, under a specific night-time
camera and a specific lighting condition:

| Stage | Script | What it learns |
| --- | --- | --- |
| 1 | `Data_Generation/train_t2i_dive.py` | LoRA on the UNet + a textual-inversion token per identity and per camera (DiVE-style baseline). |
| 2 | `Data_Generation/train_t2i_envtoken.py` | **Environment tokens** — one learned token per lighting group per camera, initialised from the stage-1 weights. |
| 3 | `Data_Generation/train_t2i_ipadapter.py` | **IP-Adapter** conditioning on OSNet identity features, so appearance is carried over from a reference image instead of only from a text token. |

Inference (`Data_Generation/generate_t2i.py`) then takes daytime Market-1501
crops as identity references and generates night-time images for every
camera / environment combination.

---

## 1. Repository layout

```
DNA-Diff/
├── environment.yml                 # conda environment (primary install path)
├── requirements.txt                # pip-only equivalent
├── Preprocess/
│   ├── group_by_lighting.py        # split NightReID by brightness into env groups
│   ├── market2nightreid.py         # append Market-1501 as the daytime camera
│   └── c2hf.py                     # convert the folder tree to a HF arrow dataset
├── Data_Generation/
│   ├── train_t2i_dive.py           # stage 1
│   ├── train_t2i_envtoken.py       # stage 2
│   ├── train_t2i_ipadapter.py      # stage 3
│   ├── lora_ipadapter.py           # IP-Adapter attention processors / projection
│   ├── generate_t2i.py             # inference
│   ├── semantic_aug/               # dataset + augmentation/pipeline wrappers
│   ├── deep_person_reid/           # vendored torchreid (OSNet) — not in git
│   └── scripts/
│       ├── finetune_step1.sh  finetune_step2.sh  finetune_step3.sh
│       ├── finetune_all.sh         # runs all three stages sequentially
│       └── sample.sh               # inference wrapper
├── move_synthetic_market1501.py    # export generated images to Market-1501 naming
└── dataset/                        # data + checkpoints — not in git
```

Everything under `dataset/`, `Data_Generation/deep_person_reid/`,
`Data_Generation/finetune_model/`, `Data_Generation/outputs/` and
`stable-diffusion-v1-5/` is git-ignored and must be obtained/produced locally.

---

## 2. Requirements

* Linux x86_64
* NVIDIA GPU(s) — training was done on **3 × 24 GB** GPUs with `batch_size=32`,
  LoRA `rank=128`, fp16
* NVIDIA driver supporting CUDA 12.1 or newer (the CUDA runtime ships with the
  PyTorch wheels; no system CUDA toolkit is required)
* conda / mamba
* ~80 GB free disk for the dataset, checkpoints and generated images

---

## 3. Installation

### 3.1 Clone the repository

```bash
git clone <this-repo> DNA-Diff
cd DNA-Diff
```

### 3.2 Create the environment

```bash
conda env create -f environment.yml
conda activate dna-diff
```

<details>
<summary>pip-only alternative</summary>

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
</details>

> `diffusers` is pinned to **0.23.1**. The training scripts subclass the LoRA
> attention processors from that release; later versions relocate them and the
> import will fail.

Configure `accelerate` once (choose multi-GPU, fp16):

```bash
accelerate config
```

### 3.3 Provide the OSNet ReID backbone

Stage 3 and inference extract identity features with OSNet from a **vendored,
locally modified** copy of
[`deep-person-reid`](https://github.com/KaiyangZhou/deep-person-reid) that lives
at `Data_Generation/deep_person_reid/`.

> **Do not replace it with a fresh upstream clone.** Some of its modules have
> been rewritten to import through the `deep_person_reid.` prefix
> (`torchreid/utils/feature_extractor.py`), while the rest still use bare
> `torchreid` imports. Upstream has neither mix, so a clean clone will not work
> as a drop-in. The directory is currently listed in `.gitignore`, so it is not
> carried by the repository — if you are setting this up on a new machine, copy
> `Data_Generation/deep_person_reid/` across from an existing checkout.

Place the **`osnet_ain_x1_0`** multi-source checkpoint (`MS -> D+M`, from the
[torchreid model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html))
at exactly:

```
Data_Generation/deep_person_reid/osnet_ain_ms_d_m.pth.tar
```

**No installation step is required.** `generate_t2i.py` and
`train_t2i_ipadapter.py` add `Data_Generation/` and
`Data_Generation/deep_person_reid/` to `sys.path` themselves, resolved relative
to the script, so the vendored copy is picked up without `pip install`.

Do **not** `pip install torchreid` from PyPI. That is an unrelated third-party
repackaging (version 0.2.5, with a different `torchreid/reid/...` layout). It
happens to satisfy the vendored package's bare `torchreid` imports, which masks
a misconfigured path and silently mixes two different copies of the library.

The OSNet checkpoint path above is **relative to the working directory**, so all
training and generation commands must still be run from inside
`Data_Generation/`.

### 3.4 Download Stable Diffusion v1.5

```bash
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
```

The scripts run with `--local_files_only`, so the model must be on disk. Note
its path — it is passed as `model_path` to every script below.

---

## 4. Data preparation

### 4.1 Download the source datasets

* **NightReID** — the night-time ReID training split (`bounding_box_train`),
  filenames of the form `0042R3_*.jpg` (`L1..L3`, `R1..R3` = 6 cameras).
* **Market-1501** (`Market-1501-v15.09.15`) — used as the daytime reference
  camera; the `gt_bbox` folder is the one that is read.

### 4.2 Split the night-time cameras into lighting groups

`Preprocess/group_by_lighting.py` sorts each camera's crops by mean HSV
*value* and splits them into `NUM_GROUPS` (default **10**) equally sized
environment groups.

This script is configured by **constants at the top of the file**, not by CLI
flags — edit them first:

```python
INPUT_PATH  = "/path/to/NightReID/bounding_box_train/"
OUTPUT_PATH = "/path/to/DNA-Diff/dataset/NightReID_Market1501"
NUM_GROUPS  = 10
```

```bash
python Preprocess/group_by_lighting.py
```

It **wipes and recreates** `OUTPUT_PATH`, then writes:

```
dataset/NightReID_Market1501/
├── cam1/env1 … env10/   <id:04d>/*.jpg   # env = (cam-1)*10 + level, 1 = darkest
├── cam2/env11 … env20/  …
├── …
├── cam6/env51 … env60/  …
└── exp/train_id.txt                       # comma-separated relabelled identities
```

### 4.3 Append Market-1501 as the daytime camera

```bash
python Preprocess/market2nightreid.py \
    --market_gt_bbox_dir  /path/to/Market-1501-v15.09.15/gt_bbox \
    --nightreid_dir       dataset/NightReID_Market1501/ \
    --max_nightreid_id    500 \
    --num_cameras_nightreid 6 \
    --num_env_tokens_per_camera 10 \
    --min_images_per_id   2
```

Market identities with at least `--min_images_per_id` images are copied to
`cam7/env61/` and relabelled starting at `max_nightreid_id + 1` (so `0501`,
`0502`, …); `exp/train_id.txt` is appended in place.

> `--max_nightreid_id` must match the highest identity produced in 4.2,
> otherwise identities collide. The same value is reused at generation time as
> `--threshold`, which tells the sampler which identities are the daytime ones.

### 4.4 Convert to a HuggingFace dataset

```bash
python Preprocess/c2hf.py \
    --dataset     NightReID_Market1501 \
    --base_path   dataset/NightReID_Market1501 \
    --num_cameras 7
```

This writes the arrow dataset **into `--base_path`** (`data-*.arrow`,
`dataset_info.json`, `state.json`) and a
`camera_environment_description_mappings_NightReID_Market1501.json` file into
the current directory, mapping each camera to a random 4-character
pseudo-word used as its text description.

Only crops with `height >= 2 * width` are kept, and identities that appear in
only one camera are dropped (there would be no cross-camera reference image).

The resulting tree is what all later steps point at:

```
dataset/NightReID_Market1501/
├── cam1 … cam6/env*/      # night-time, 10 environments each
├── cam7/env61/            # daytime (Market-1501) reference images
├── exp/train_id.txt
├── data-0000*-of-*.arrow, dataset_info.json, state.json
└── finetune_model/        # created by training
```

---

## 5. Training

All three stages read the dataset through the `HUG_LOCAL_IMAGE_TRAIN_DIR`
environment variable, which the shell wrappers set for you. Run everything from
`Data_Generation/`.

### 5.1 Run the whole pipeline

```bash
cd Data_Generation
GPU_IDS="0,1,2" bash scripts/finetune_all.sh \
    /abs/path/to/DNA-Diff/dataset/NightReID_Market1501/ \
    /abs/path/to/stable-diffusion-v1-5
```

Positional arguments: `data_dir  model_path  steps  batchsize  nepoch  rank  gpu_ids`
(defaults: `"1 2 3"`, `32`, `100`, `128`, `1,2,3`).

Useful environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `STEPS` | `"1 2 3"` | which stages to run, e.g. `"2 3"` |
| `GPU_IDS` | `1,2,3` | physical GPU ids passed to `accelerate --gpu_ids` |
| `LOG_DIR` | `<data_dir>/finetune_model/logs` | per-stage logs |
| `STEP1_MAX_STEPS` / `STEP1_CKPT_STEPS` | `100000` / `25000` | stage-1 schedule |
| `STEP2_MAX_STEPS` / `STEP2_CKPT_STEPS` | `40000` / `10000` | stage-2 schedule |
| `STEP3_MAX_STEPS` / `STEP3_CKPT_STEPS` | `30000` / `10000` | stage-3 schedule |

Do **not** also set `CUDA_VISIBLE_DEVICES` — `GPU_IDS` is already physical
indices and setting both remaps them twice.

The wrapper verifies that each stage produced the artefacts the next stage
needs, and logs to `<data_dir>/finetune_model/logs/stepN_<timestamp>.log`.

### 5.2 Run a single stage

```bash
cd Data_Generation
source scripts/finetune_step1.sh      # defines finetune()
GPU_IDS="0,1,2" finetune \
    /abs/path/to/dataset/NightReID_Market1501/ \
    /abs/path/to/stable-diffusion-v1-5
```

`finetune` takes
`data_dir  model_path  output_dir  max_steps  batchsize  nepoch  rank  checkpointing_steps  gpu_ids`.
The same applies to `finetune_step2.sh` and `finetune_step3.sh`.

### 5.3 Outputs

```
dataset/NightReID_Market1501/finetune_model/
├── finetune_stage1_dive/
│   ├── checkpoint-100000/model.safetensors
│   └── learned_embeds-steps-100000.bin
├── finetune_stage2_envtoken/
│   ├── checkpoint-40000/model.safetensors
│   └── learned_embeds-steps-40000.bin
└── finetune_stage3_ipadapter/
    └── checkpoint-30000/…
```

### 5.4 ⚠️ Cross-stage checkpoint paths are hard-coded

Each stage loads the previous stage's weights from paths that embed the
previous stage's **step count**:

| File | Line | Expects |
| --- | --- | --- |
| `train_t2i_envtoken.py` | 507, 593 | `finetune_stage1_dive/learned_embeds-steps-100000.bin`, `finetune_stage1_dive/checkpoint-100000/model.safetensors` |
| `train_t2i_ipadapter.py` | 537, 593, 637 | `finetune_stage1_dive/learned_embeds-steps-100000.bin`, `finetune_stage2_envtoken/learned_embeds-steps-40000.bin`, `finetune_stage2_envtoken/checkpoint-40000/model.safetensors` |
| `generate_t2i.py` | 81, 83 | `finetune_stage3_ipadapter/checkpoint-30000`, `finetune_stage1_dive/learned_embeds-steps-100000.bin` |
| `semantic_aug/datasets/person_revised.py` | 512 | `finetune_stage1_dive/learned_embeds-steps-100000.bin` |

If you change `STEP*_MAX_STEPS`, update these paths too. `finetune_all.sh`
prints a warning when the schedules and the hard-coded paths diverge, but it
cannot fix them.

---

## 6. Generation

```bash
cd Data_Generation
source scripts/sample.sh
sample \
    /abs/path/to/dataset/NightReID_Market1501/ \
    /abs/path/to/dataset/NightReID_Market1501/cam7/env61/ \
    /abs/path/to/stable-diffusion-v1-5 \
    /abs/path/to/output_dir \
    10 512 false
```

`sample` takes
`data_dir  cond_daytime_path  model_path  output_dir  num_env_tokens_per_camera  resolution  use_pretrained_weights`.
`cond_daytime_path` is the **daytime camera folder** (`cam7/env61/`) whose
crops provide the OSNet identity embeddings.

Key `generate_t2i.py` flags (edit them in `scripts/sample.sh`):

| Flag | Default in `sample.sh` | Meaning |
| --- | --- | --- |
| `--cameras` | `cam1 … cam6` | target night-time cameras |
| `--gpu_ids` | `1 2 3` | one worker process per GPU |
| `--batch_size` | `60` | queue items pulled per pipeline call |
| `--repeat_each_id` | `3` | how many times each identity is enqueued per camera |
| `--images_per_call` | `1` | images generated per pipeline call |
| `--guidance_scale` | `7.5` | classifier-free guidance strength |
| `--threshold` | `500` | identities with numeric id ≤ this are ignored |
| `--resolution` | `512` | output size |
| `--debug` | off | single-process mode, easier to trace |

`--lora_path` and `--embed_path` are accepted but unused — the LoRA checkpoint
and the learned embeddings are resolved from `HUG_LOCAL_IMAGE_TRAIN_DIR` (see
§5.4).

Images are written as:

```
<output_dir>/<camN>/<id:04d>/<env_id>_<n>.jpg
```

---

## 7. Exporting synthetic data for ReID training

Flatten the generated tree into Market-1501 filename conventions
(`<id+10000:05d>_c<cam>_<orig>.jpg`, 3 images per camera per identity):

```bash
mkdir -p /path/to/market_style_output
python move_synthetic_market1501.py \
    -i /abs/path/to/output_dir \
    -o /path/to/market_style_output
```

The result can be merged into a Market-1501 `bounding_box_train` folder and
used to train any downstream ReID model.

---

## 8. Paths to change when porting to a new machine

Besides the hard-coded checkpoint paths in §5.4, the following absolute paths
are baked into the source and must be edited:

| File | Line | Path |
| --- | --- | --- |
| `Data_Generation/semantic_aug/augmentations/dreabooth_lora_mt.py` | 2, 24 | `sys.path.append(...)` |
| `Preprocess/group_by_lighting.py` | 22–23 | `INPUT_PATH`, `OUTPUT_PATH` |
| `Preprocess/market2nightreid.py` | 15, 21 | argparse defaults (overridable on the CLI) |
| `Data_Generation/scripts/*.sh` | — | `data_dir`, `model_path` defaults (overridable as arguments) |

---

## 9. Troubleshooting

**`ValueError: HUG_LOCAL_IMAGE_TRAIN_DIR env variable is not set.`**
You ran a training/generation script directly. Use the wrappers in
`Data_Generation/scripts/`, or `export HUG_LOCAL_IMAGE_TRAIN_DIR=/abs/path/to/dataset/NightReID_Market1501/`.

**`FileNotFoundError: deep_person_reid/osnet_ain_ms_d_m.pth.tar`**
The OSNet path is relative — run from `Data_Generation/`, and check §3.3.

**`ModuleNotFoundError: No module named 'torchreid'`**
Raised from `deep_person_reid/torchreid/__init__.py`, which imports `torchreid`
absolutely. It means `Data_Generation/deep_person_reid/` is not on `sys.path`.
The two scripts that use OSNet add it themselves, so this normally only appears
if you run them from outside `Data_Generation/` or import them from your own
code. Fix it with
`export PYTHONPATH="$PWD:$PWD/deep_person_reid:$PYTHONPATH"` from
`Data_Generation/`. Installing PyPI `torchreid` also makes the error go away,
but it is the wrong package — see §3.3.

**`ImportError` from `diffusers.models.attention_processor`**
`diffusers` is not at 0.23.1. Reinstall with `pip install diffusers==0.23.1`.

**Stage 2 or 3 fails loading the previous stage's checkpoint**
The step counts in the hard-coded paths do not match what you trained. See §5.4.

**`accelerate` uses the wrong GPUs**
`GPU_IDS` and `CUDA_VISIBLE_DEVICES` are both set. Use only `GPU_IDS`.

**OOM during training**
Lower `batchsize` (arg 5 of `finetune`) and/or `rank` (arg 7); `rank=128` with
`batchsize=32` needs roughly 24 GB per GPU.
