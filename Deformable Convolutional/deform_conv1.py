import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

'''
在卷积前加入一层filter(offset)学习下层卷积核的位置偏移
'''


class Conv_Offset(nn.Conv2d):

    def __init__(self, channels, init_normal_stddev=0.01, **kwargs):
        # super(Conv_Offset, self).__init__(channels, channels * 2, 3, padding=1, bias=False, **kwargs)
        # self.weight.data.copy_(self._init_weights(self.weight,init_normal_stddev))
        super(Conv_Offset, self).__init__(channels, channels * 2,**kwargs)
        self._grid_param = None

    def _get_grid(self, x):

        def np_repeat_2d(a, repeats):
            """Tensorflow version of np.repeat for 2D"""

            assert len(a.shape) == 2
            a = np.expand_dims(a, 0)
            a = np.tile(a, [repeats, 1, 1])
            return a

        def th_generate_grid(batch_size, input_height, input_width, dtype, cuda):
            grid = np.meshgrid(
                range(input_height), range(input_width), indexing='ij'
            )
            grid = np.stack(grid, axis=-1)
            grid = grid.reshape(-1, 2)

            grid = np_repeat_2d(grid, batch_size)
            grid = torch.from_numpy(grid).type(dtype)
            if cuda:
                grid = grid.cuda()
            return Variable(grid, requires_grad=False)

        batch_size, input_height, input_width = x.size(0), x.size(1), x.size(2)
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_param == (batch_size, input_height, input_width, dtype, cuda):
            return self._grid
        self._grid_param = (batch_size, input_height, input_width, dtype, cuda)
        self._grid = th_generate_grid(batch_size, input_height, input_width, dtype, cuda)
        return self._grid

    def th_batch_map_coordinates(self, input, coords, order=1):

        def th_flatten(a):
            """Flatten tensor"""
            return a.contiguous().view(a.nelement())

        def th_repeat(a, repeats, axis=0):
            """Torch version of np.repeat for 1D"""
            assert len(a.size()) == 1
            return th_flatten(torch.transpose(a.repeat(repeats, 1), 0, 1))

        """Batch version of th_map_coordinates
        Only supports 2D feature maps
        Parameters
        ----------
        input : tf.Tensor. shape = (b, s, s)
        coords : tf.Tensor. shape = (b, n_points, 2)
        Returns
        -------
        tf.Tensor. shape = (b, s, s)
        """

        batch_size = input.size(0)
        input_height = input.size(1)
        input_width = input.size(2)

        n_coords = coords.size(1)

        # coords = torch.clamp(coords, 0, input_size - 1)

        coords = torch.cat((torch.clamp(coords.narrow(2, 0, 1), 0, input_height - 1),
                            torch.clamp(coords.narrow(2, 1, 1), 0, input_width - 1)), 2)

        assert (coords.size(1) == n_coords)

        coords_lt = coords.floor().long()
        coords_rb = coords.ceil().long()
        coords_lb = torch.stack([coords_lt[..., 0], coords_rb[..., 1]], 2)
        coords_rt = torch.stack([coords_rb[..., 0], coords_lt[..., 1]], 2)
        idx = th_repeat(torch.arange(0, batch_size), n_coords).long()
        idx = Variable(idx, requires_grad=False)
        if input.is_cuda:
            idx = idx.cuda()

        def _get_vals_by_coords(input, coords):
            indices = torch.stack([
                idx, th_flatten(coords[..., 0]), th_flatten(coords[..., 1])
            ], 1)
            inds = indices[:, 0] * input.size(1) * input.size(2) + indices[:, 1] * input.size(2) + indices[:, 2]
            vals = th_flatten(input).index_select(0, inds)
            vals = vals.view(batch_size, n_coords)
            return vals

        vals_lt = _get_vals_by_coords(input, coords_lt.detach())
        vals_rb = _get_vals_by_coords(input, coords_rb.detach())
        vals_lb = _get_vals_by_coords(input, coords_lb.detach())
        vals_rt = _get_vals_by_coords(input, coords_rt.detach())

        coords_offset_lt = coords - coords_lt.type(coords.data.type())
        vals_t = coords_offset_lt[..., 0] * (vals_rt - vals_lt) + vals_lt
        vals_b = coords_offset_lt[..., 0] * (vals_rb - vals_lb) + vals_lb
        mapped_vals = coords_offset_lt[..., 1] * (vals_b - vals_t) + vals_t
        return mapped_vals

    def th_batch_map_offsets(self, input, offsets, order=1):
        '''

        :param input: bc,h,w
        :param offsets: bc,h,w,2
        :param grid:
        :param order:
        :return: bc,h,w
        '''

        b, h, w = input.size()
        offsets = offsets.view(b, -1, 2)
        coords = offsets + self._get_grid(input)
        mapped_vals = self.th_batch_map_coordinates(input, coords)


        return mapped_vals

    def forward(self, x):

        shape = x.size()
        offset = super(Conv_Offset, self).forward(x)

        # 2: bc,h,w,2
        offset = offset.contiguous().view(-1, int(shape[2]), int(shape[3]), 2)

        # bc,h,w
        x = x.contiguous().view(-1, int(shape[2]), int(shape[3]))

        offset = self.th_batch_map_offsets(x, offset)

        offset = offset.contiguous().view(-1, int(shape[1]), int(shape[2]), int(shape[3]))

        return offset


class DCNv1(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernerl=3, stride=1, padding=1, groups=1, dilation = 1,bias = False,is_relu = True,is_bn = True):
        super(DCNv1, self).__init__(in_channels=in_channels, out_channels=out_channels,kernel_size=kernerl,
                                    stride=stride, padding=padding, groups=groups, dilation = dilation,bias = bias)

        self.offset = Conv_Offset(in_channels,kernel_size=kernerl,stride=stride,
                                  padding=padding,groups=groups,dilation=dilation,bias = bias)
        self.is_bn = is_bn
        self.is_relu = is_relu
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):

        x = self.bn(x) if self.is_bn else x
        offset = self.offset(x)
        x = super(DCNv1, self).forward(offset)
        return self.relu(x) if self.is_relu else x




if __name__ == '__main__':

    x = torch.randn(1, 3, 20, 20)

    dcn = DCNv1(3,64,dilation=3,padding=3)
    # x_offset = Conv_Offset(3)(x)
    d = dcn(x)
    print(d)

    print(d.shape)

    c = nn.Conv2d(3,64,3,padding=1)(x)
    print(x)




