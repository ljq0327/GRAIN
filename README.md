# GRAIN

This project provides action recognition training code.

## Requirements

- Python 3.8
- PyTorch + CUDA (multi-GPU supported via `--gpu`)

## Training

### Launch Command

Run training with:

```bash
python main.py --train_classifier --gpu 0,1 --run_id numacv --dataset numa --training_mode CV --model_version cross_view --batch_size 8 --num_epochs 100 --num_workers 8 --validation_mode multi_view --learning_rate 3e-5 --weight_decay 1e-6 --optimizer ADAM --validation_interval 3 --seed 42
```

### Dataset Options

The `--dataset` argument supports:

- `ntu_rgbd_60`
- `ntu_rgbd_120`
- `pkummd`
- `numa`

Example:

```bash
python main.py --train_classifier --gpu 0,1 --run_id exp1 --dataset ntu_rgbd_60 --training_mode CV --model_version cross_view --batch_size 16 --num_epochs 100 --num_workers 8 --validation_mode multi_view --learning_rate 4e-5 --weight_decay 1e-6 --optimizer ADAM --validation_interval 3 --seed 42
```


