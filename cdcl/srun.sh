#! /bin/bash
srun --cpus-per-task=2 --mem=40GB --time=04:00:00 --gres=gpu:1 --pty /bin/bash