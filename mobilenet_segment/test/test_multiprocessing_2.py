import torch
from torch.multiprocessing import Pool, Process, set_start_method
from torch.autograd import Variable
import numpy as np
from scipy.ndimage import zoom

def get_pred(args):

  img = args[0]
  scale = args[1]
  # feed input data
  input_img = Variable(torch.from_numpy(img),
                     volatile=True).cuda()
  return input_img

if __name__ == '__main__':
    try:
     set_start_method('spawn')
    except RuntimeError:
        pass

    img = np.float32(np.random.randint(0, 2, (300, 300, 3)))
    scales = [1,2,3,4,5]
    scale_list = []
    for scale in scales: 
        scale_list.append([img,scale])
    multi_pool = Pool(processes=5)
    predictions = multi_pool.map(get_pred,scale_list)
    multi_pool.close() 
    multi_pool.join()