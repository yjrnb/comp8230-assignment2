# comp8230-assignment2
This is the code for *Assignment 2 of COMP8230 Session 1 2026 by Jinrui Yang*, implementing medical image synthesis (particularly brain MRI) with **Flow Matching**.

## Loading Model Checkpoint
Download the model checkpoint in this [link](https://drive.google.com/file/d/1w4ov4rKf2noLNzNbOxlWNyVg2HclvmmO/view?usp=drive_link) to the folder `./models`.

## Generating Synthetic Data
Use `sampling.py` to generate synthetic samples from a model checkpoint and save them as a `.pkl`:
```bash
python sampling.py \
    --config_path configs/unconditional.yaml \
    --model_path models/unconditional_checkpoints_3d_mri/default \
    --num_samples 200
```