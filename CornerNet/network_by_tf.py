import tensorflow as tf
from corner_pooling import TopPool,BottomPool, LeftPool, RightPool

class Corner:

    def __init__(self,is_train):

        self.is_train = is_train
        self.n_deep = 5
        self.n_dims = [256, 256, 384, 384, 384, 512]
        self.n_ksize = [2, 2, 2, 2, 2, 4]



    def head(self,img):

        with tf.variable_scope('head'):

            net = tf.layers.conv2d(img,128,7,2,padding='same',name='conv1')
            net = tf.layers.batch_normalization(net,trainable=self.is_train,name='bn1')
            net = self.residual(net,256,strides=2,name='residual_start')  #128,128,256

            return net


    def conv_bn_relu(self,x,filters,k_size = 3,strides = 1,is_relu = True,is_bn = True,name = 'conv_bn_re'):

        with tf.variable_scope(name):

            net = tf.layers.conv2d(x,filters,k_size,strides=strides,padding='same',name='res_conv')
            if is_bn:
                net = tf.layers.batch_normalization(net,trainable=self.is_train,name='res_bn')
            if is_relu:
                net = tf.nn.relu(net)

            return net


    def residual(self,x,filters,k_size = 3,strides = 1,name = 'residual'):
        with tf.variable_scope(name):
            net = self.conv_bn_relu(x,filters,k_size=k_size,strides=strides,name = 'up_1')
            net = self.conv_bn_relu(net,filters,k_size=k_size,is_relu=False,name = 'up_2')

            #skip  #1x1 conv
            skip = self.conv_bn_relu(x,filters = filters,k_size= 1,strides=strides,is_relu=False,name='low')

            out = tf.nn.relu(tf.add(skip,net))

            return out

    def res_block(self,x,filters,k_size = 3,iter = 1,name = 'res_block'):

        with tf.variable_scope(name):

            net = self.residual(x,filters,k_size,name='residual_0')
            for i in range(1,iter):
                net = self.residual(net,filters,k_size,name='residual_%d' % i)
            return net

    def hourglass_net(self,x,deep,k_size,filters,name = 'hourglass_iter'):

        with tf.variable_scope(name):
            k1 = k_size[0]
            k2 = k_size[1]
            curr_filter = filters[0]
            next_filter = filters[1]

            up1 = self.res_block(x,curr_filter,k_size=k1,name='up_1')
            net = tf.layers.max_pooling2d(x,(2,2),(2,2),padding='same')

            down_1 = self.res_block(net,next_filter,k1,name='down_1')
            if deep > 1:
                down_2 = self.hourglass_net(down_1,deep - 1,k_size=k_size[1:],filters=filters[1:],name='hourglass_%d' % (deep - 1))
            else:
                down_2 = self.res_block(down_1,filters=next_filter,k_size=k2,name='down_2')
            down_3 = self.res_block(down_2,curr_filter,k_size=k1,name='down_3')

            up2 = tf.image.resize_nearest_neighbor(down_3,tf.shape(down_3)[1:3] * 2,name='up_2')

            out = tf.add(up1,up2)

            return out

    def corner_pool(self,x,filters,k_size = 3,name = 'corner_pooling'):
        with tf.variable_scope(name):

            with tf.variable_scope('top_left'):

                top_ponits = self.conv_bn_relu(x,128,name = 'top')
                top_pool = TopPool(top_ponits)

                left_points = self.conv_bn_relu(x,128,name='left')
                left_pool = LeftPool(left_points)

                top_left = tf.add(top_pool,left_pool)
                top_left = self.conv_bn_relu(top_left,filters=filters,is_relu=False,name='_top_left')
                _top_left = self.conv_bn_relu(x,filters=filters,is_relu=False,k_size=1,name='skip')
                top_left = tf.add(top_left,_top_left)
                top_left = tf.nn.relu(top_left)
                top_left = self.conv_bn_relu(top_left,filters = filters,name='top_left')


            with tf.variable_scope('bottom_right'):

                bottom_poinst = self.conv_bn_relu(x,128,name='bottom')
                bottom_pool = BottomPool(bottom_poinst)

                right_points = self.conv_bn_relu(x,128,name='right')
                right_pool = RightPool(right_points)

                bottom_right = tf.add(bottom_pool,right_pool)
                bottom_right = self.conv_bn_relu(bottom_right,filters = filters,is_relu=False,name='_bottom_right')
                _bottom_right = self.conv_bn_relu(x,filters=filters,is_relu=False,k_size=1,name='skip')

                bottom_right = tf.add(bottom_right,_bottom_right)
                bottom_right = tf.nn.relu(bottom_right)
                bottom_right = self.conv_bn_relu(bottom_right,filters=filters,name='bottom_right')

                return top_left,bottom_right

    def heatmap(self,x,filters,name = 'heatmaps'):
        #channel:80
        with tf.variable_scope(name):
            net = self.conv_bn_relu(x,filters=filters[0],is_bn=False,name = 'conv1')
            net = tf.layers.conv2d(net,filters=filters[1],kernel_size=1,name='conv2')

            return net

    def embedding(self,x,filters,name = 'embedding'):

        #channel:1
        with tf.variable_scope(name):
            net = self.conv_bn_relu(x,filters[0],is_bn=False,name='conv1')
            net = tf.layers.conv2d(net,filters[1],1,name='conv2')

            return net

    def offset(self,x,filters,name = 'offset'):
        #channel : 2
        with tf.variable_scope(name):
            net = self.conv_bn_relu(x, filters[0], is_bn=False, name='conv1')
            net = tf.layers.conv2d(net, filters[1], 1, name='conv2')

            return net

    def inter(self,x1,x2,filters,name = 'inter'):
        with tf.variable_scope(name):
            net1 = self.conv_bn_relu(x1,filters=filters,is_relu=False,k_size=1,name='branch_start')
            net2 = self.conv_bn_relu(x2,filters=filters,is_relu=False,k_size=1,name='branch_hourglass2')
            net = tf.nn.relu(tf.add(net1,net2))
            net = self.residual(net,filters=filters)
            return net

    def main_net(self,img,
                 gt_tl_embedding = None,
                 gt_br_embedding = None):
        with tf.variable_scope('corner_net'):

            start_net = self.head(img)  #[,128,128,256]
            print(start_net)
            with tf.variable_scope('hourglass_1'):

                #[,128,128,256]
                hg1 = self.hourglass_net(start_net,deep=self.n_deep,k_size=self.n_ksize,filters=self.n_dims,name='hg_1')

                hinge1 = self.conv_bn_relu(hg1,filters=256,name='hinge1')
                top_left,bottom_right = self.corner_pool(hinge1,256)


                #top_left的heatmap和embedding
                tl_heatmap = self.heatmap(top_left,[256,80],name='top_left_heatmap1')
                tl_embedding = self.embedding(top_left,[256,1],name='top_left_embedding1')
                tl_offset = self.offset(top_left,[256,2],name='top_left_offset1')

                #bottom_right
                br_heatmap = self.heatmap(bottom_right, [256, 80], name='bottom_right_heatmap1')
                br_embedding = self.embedding(bottom_right, [256, 1], name='bottom_right_embedding1')
                br_offset = self.offset(bottom_right, [256, 2], name='bottom_right_offset1')

            with tf.variable_scope('hourglass_2'):
                inter = self.inter(start_net,hinge1,256)
                hg2 = self.hourglass_net(inter,deep=self.n_deep,k_size=self.n_ksize,filters=self.n_dims,name='hg_2')
                hinge2 = self.conv_bn_relu(hg2,filters=256,name='hinge2')
                top_left2,bottom_right2 = self.corner_pool(hinge2,256)

                tl_heatmap2 = self.heatmap(top_left2,[256,80],name='top_left_heatmap2')
                tl_embedding2 = self.embedding(top_left2, [256, 1], name='top_left_embedding2')
                tl_offset2 = self.offset(top_left2, [256, 2], name='top_left_offset2')
                br_heatmap2 = self.heatmap(bottom_right2, [256, 80], name='bottom_right_heatmap2')
                br_embedding2 = self.embedding(bottom_right2, [256, 1], name='bottom_right_embedding2')
                br_offset2 = self.offset(bottom_right2, [256, 2], name='bottom_right_offset2')

        if self.is_train:
            return [tl_heatmap,br_heatmap,tl_embedding,br_embedding,tl_offset,br_offset,tl_heatmap2,br_heatmap2,tl_embedding2,br_embedding2,tl_offset2,br_offset2]
        return [tl_heatmap2,br_heatmap2,tl_embedding2,br_embedding2,tl_offset2,br_offset2]



if __name__ == '__main__':
    x = tf.placeholder(dtype=tf.float32,shape=[None,511,511,3])
    test1 = Corner(True)
    test_res = test1.main_net(x)
    print(test_res)