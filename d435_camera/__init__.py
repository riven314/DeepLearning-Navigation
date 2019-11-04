import os
import sys

crt_dir = os.getcwd()
this_dir = os.path.join(crt_dir, 'd435_module')
# when this folder serve as a module 
if os.path.isdir(this_dir):
    sys.path.append(this_dir)
    print('d435_module added to python path: {}'.format(this_dir))