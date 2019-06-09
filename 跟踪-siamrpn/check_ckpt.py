import numpy as np
from tensorflow.python import pywrap_tensorflow

# checkpoint_path = '../Nn/ssd_vgg_300_weights.ckpt'
# checkpoint_path = '../NN/ssd_vgg_300_weights.ckpt'
# checkpoint_path = './model/lanenet/tusimple_lanenet.ckpt'
# checkpoint_path = 'MTCNN/tensorflow_version/ONet/ONet-14'

checkpoint_path = 'ckpt/pre-train/'
# print(checkpoint_path)
# read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

data_print = np.array([])
for key in var_to_shape_map:
    print('tensor_name', key)
    ckpt_data = np.array(reader.get_tensor(key))  # cast list to np arrary
    print(ckpt_data.shape)
    ckpt_data = ckpt_data.flatten()  # flatten list
    data_print = np.append(data_print, ckpt_data, axis=0)

# print(data_print, data_print.shape, np.max(data_print), np.min(data_print), np.mean(data_print))
print(data_print)