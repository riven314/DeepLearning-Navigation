import os
import sys

path = os.path.join(os.getcwd(), 'mobilenet_segment')
if os.path.isdir(path) and path not in sys.path:
    sys.path.append(path)
    print('path added: {}'.format(path))