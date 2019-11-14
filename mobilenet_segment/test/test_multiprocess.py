"""
test the following:
1. speed improvement WITH torch.multiprocessing
2. for each individual process, input numpy array and transform to tensor

REFERENCE:
1. multiprocessing not work in IDLE: https://stackoverflow.com/questions/40453332/multiprocessing-process-not-working-properly
2. fail CUDA IPC operation: https://pytorch.org/docs/stable/notes/windows.html#cuda-ipc-operations
3. build pytorch from source: https://www.youtube.com/watch?v=sGWLjbn5cgs
4. build pytorch from source in ubuntu: https://medium.com/repro-repo/build-pytorch-from-source-on-ubuntu-18-04-1c5556ca8fbf

OBSERVATION:
1. within a process: CPU to GPU --> error
"""
import numpy as np
import torch
import torch.multiprocessing as mp
from functools import partial

NUM_PROCESS = 5

def single(b):
    device = torch.device('cuda', 0)
    np_array = np.random.randint(0, 2, (200, 200, 3))
    np_array = np.float32(np_array)
    x = torch.from_numpy(np_array.copy()).to(device)
    #x = torch.from_numpy(np_array)
    out = b * x 
    return out

def hihi(t, b):
    return b * t

def parallel_hihi():
    np_array = np.random.randint(0, 2, (200, 200, 3))
    np_array = np.float32(np_array)
    device = torch.device('cuda', 0)

    t = torch.from_numpy(np_array).to(device)
    t.share_memory_()
    b = [100., 200., 300.]
    num_processes = 3
    processes = []
    for j in b:
        p = mp.Process(target = hihi, args=(t, b))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    return p

def parallel_single():
    np_array = np.random.randint(0, 2, (200, 200, 3))
    np_array = np.float32(np_array)
    a = 3
    b = [100., 200., 300.]

    pool = mp.Pool()
    #func = partial(single)
    #k = pool.map(func, b)
    k = pool.map(single, b)
    pool.close()
    pool.join()
    return k

if __name__ == '__main__':
    k = parallel_hihi()