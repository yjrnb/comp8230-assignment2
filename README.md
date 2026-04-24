# Accelerating Medical Image Synthesis with Flow Matching
This is the code for *Assignment 2 of COMP8230 Session 1 2026 by Jinrui Yang*, implementing medical image synthesis (particularly brain MRI) with **Flow Matching**.

## Loading Model Checkpoint
Download the model checkpoint in this [link](https://drive.google.com/file/d/1w4ov4rKf2noLNzNbOxlWNyVg2HclvmmO/view?usp=drive_link) to the folder `./models`.


## Requirements
`uv` is used for managing Python packages. To set up a virtual environment, please run `uv sync` on the root directory of this project.


## Generating Synthetic Data
Use `sampling.py` to generate synthetic samples from a model checkpoint and save them as a `.pkl`:
```bash
python sampling.py \
    --config_path configs/unconditional.yaml \
    --model_path models/unconditional_checkpoints_3d_mri/default \
    --num_samples 200
```

### Arguments
Arguments
- `--config_path` (`str`, default: `configs/default.yaml`): Config file used for model/data setup.
- `--model_path` (`str`, optional): Checkpoint .ckpt file or directory.
- `--num_samples` (`int`, optional): Number of samples to save. If omitted, saves all validation samples.
- `--num_inference_steps` (`int`, optional): Number of solver time points used during sampling. If omitted, uses solver_args.time_points from the config.
- `--output_path` (`str`, optional): Explicit output .pkl path.
- `--overwrite` (`flag`): Overwrite an existing file at --output_path.
- `--output_norm` (`str`, default: `per_sample_minmax`): One of clip_0_1, per_sample_minmax, global_minmax, none.
- `--allow_config_mismatch` (`flag`): Allow loading a checkpoint whose saved critical model fields differ from current config.
- `--seed` (`int`, optional): Override RNG seed for reproducible inference. Defaults to train_args.seed if provided.