import torch.multiprocessing as mp

from train import launch_training


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    launch_training()