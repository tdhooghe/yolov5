# %%
import torch


# import tensorflow as tf
# %%
# setting device on GPU if available, else CPU
def print_gpu_info():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')


# %%
#  for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done
# print(tf.config.list_physical_devices('GPU'))

if __name__ == "__main__":
    print_gpu_info()
