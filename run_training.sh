torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=101 --rdzv_endpoint="localhost:5972" main_training.py --model deepvit
