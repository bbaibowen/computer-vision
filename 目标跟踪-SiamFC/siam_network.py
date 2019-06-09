import tensorflow as tf
import numpy as np
import cv2

'''
进行尺度缩放
'''

contextAmount = 0.5
NUM_SCALES = 3
scaleStep = 1.0375
BATCH = 10

class Siam_Network:

    def __init__(self,is_train):
        self.is_train = is_train
        self.target_holder = tf.placeholder(dtype=tf.float32, shape=[1, 127, 127, 3])
        self.search_holder = tf.placeholder(dtype=tf.float32, shape=[NUM_SCALES, 255, 255, 3])
        self.target_holder_ = tf.placeholder(dtype=tf.float32, shape=[BATCH, 127, 127, 3])
        self.search_holder_ = tf.placeholder(dtype=tf.float32, shape=[BATCH, 255, 255, 3])
        self.train_network(self.target_holder_,self.search_holder_)


    def getVariable(self, name, shape, initializer, weightDecay = 0.0, dType=tf.float32, trainable = True):
        if weightDecay > 0:
            regularizer = tf.contrib.layers.l2_regularizer(weightDecay)
        else:
            regularizer = None

        return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dType, regularizer=regularizer, trainable=trainable)


    def conv2d(self,x,filters,k_size,strides,iter):

        channels = int(x.get_shape()[-1])
        lambda_conv = lambda i,k:tf.nn.conv2d(i,k,strides = [1,strides,strides,1],padding='VALID')

        with tf.variable_scope('conv'):

            w = self.getVariable('weights', shape=[k_size, k_size, channels / iter, filters], initializer=tf.truncated_normal_initializer(stddev=0.03), weightDecay=5e-04, dType=tf.float32, trainable=True)
            b = self.getVariable('biases', shape=[filters, ], initializer=tf.constant_initializer(value=0.1, dtype=tf.float32), weightDecay=0, dType=tf.float32, trainable=True)

        if iter == 1:
            net = lambda_conv(x,w)
        else:
            x_n = tf.split(x,iter,axis=3)
            w_n = tf.split(w,iter,axis=3)
            conv_n = [lambda_conv(i,k) for i,k in zip(x_n,w_n)]

            net = tf.concat(conv_n,axis=3)

        net = tf.nn.bias_add(net,b)

        return net

    def BN(self,x):
        shape = x.get_shape()
        paramsShape = shape[-1:]

        axis = list(range(len(shape) - 1))

        with tf.variable_scope('bn'):
            beta = self.getVariable('beta', paramsShape, initializer=tf.constant_initializer(value=0, dtype=tf.float32))

            gamma = self.getVariable('gamma', paramsShape,
                                     initializer=tf.constant_initializer(value=1, dtype=tf.float32))


        mean, variance = tf.nn.moments(x, axis)

        # mean, variance = control_flow_ops.cond(self.is_train, lambda : (mean, variance), lambda : (movingMean, movingVariance))

        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, variance_epsilon=0.001)

        return x




    def build_target_layer(self,x):

        # with tf.variable_scope('siamese') as scope:

        with tf.variable_scope('scala1'):

            net = self.conv2d(x,96,11,2,1)
            net = self.BN(net)
            net = tf.nn.relu(net)
            net = tf.nn.max_pool(net,[1,3,3,1],[1,2,2,1],padding='VALID')

        with tf.variable_scope('scala2'):

            net = self.conv2d(net,256,5,1,2)
            net = self.BN(net)
            net = tf.nn.relu(net)
            net = tf.nn.max_pool(net,[1,3,3,1],[1,2,2,1],padding='VALID')

        with tf.variable_scope('scala3'):
            net = self.conv2d(net,384,3,1,1)
            net = self.BN(net)
            net = tf.nn.relu(net)

        with tf.variable_scope('scala4'):
            net = self.conv2d(net,384,3,1,2)
            net = self.BN(net)
            net = tf.nn.relu(net)

        with tf.variable_scope('scala5'):
            net = self.conv2d(net,256,3,1,2)

        return net

    def test_network(self,search,zFeat):

        with tf.variable_scope('siamese') as scope:
            scope.reuse_variables()
            search_layer = self.build_target_layer(search)


        batchScore = int(search_layer.get_shape()[0])

        scores = tf.split(search_layer,batchScore,axis=0)
        scores1 = []
        for i in range(batchScore):
            scores1.append(tf.nn.conv2d(scores[i],zFeat,strides=[1,1,1,1],padding='VALID'))

        scores = tf.concat(scores1,axis=0)

        with tf.variable_scope('adjust') as scope:
            scope.reuse_variables()
            w = self.getVariable('weights', [1, 1, 1, 1],
                                       initializer=tf.constant_initializer(value=0.001, dtype=tf.float32),
                                       weightDecay=1.0 * 5e-04, dType=tf.float32, trainable=True)
            b = self.getVariable('biases', [1, ],
                                      initializer=tf.constant_initializer(value=0, dtype=tf.float32),
                                      weightDecay=1.0 * 5e-04, dType=tf.float32, trainable=True)

            scores = tf.nn.conv2d(scores,w,[1,1,1,1],padding='VALID')
            scores = tf.nn.bias_add(scores,b)


        return scores


    def train_network(self,target,search):

        with tf.variable_scope('siamese') as scope:
            aFeat = self.build_target_layer(target)
            scope.reuse_variables()
            score = self.build_target_layer(search)

        with tf.variable_scope('score'):
            aFeat = tf.transpose(aFeat,[1,2,3,0])
            batchAFeat = int(aFeat.get_shape()[-1])
            batchScore = int(score.get_shape()[0])
            print(batchAFeat,batchScore)

            conv = lambda i,k:tf.nn.conv2d(i,k,strides=[1,1,1,1],padding='VALID')

            aFeats = tf.split(aFeat,batchAFeat,axis=3)
            scores = tf.split(score,batchScore,axis=0)
            scores = [conv(i,k) for i,k in zip(scores,aFeats)]
            scores = tf.concat(scores,axis=3)
            scores = tf.transpose(scores,[3,1,2,0])

        with tf.variable_scope('adjust'):
            weights = self.getVariable('weights', [1, 1, 1, 1], initializer=tf.constant_initializer(value=0.001, dtype=tf.float32), weightDecay=1.0 * 5e-04, dType=tf.float32, trainable=True)
            biases = self.getVariable('biases', [1,], initializer=tf.constant_initializer(value=0, dtype=tf.float32), weightDecay=1.0 * 5e-04, dType=tf.float32, trainable=True)

            score = tf.nn.conv2d(scores,weights,strides=[1,1,1,1],padding='VALID')

            score = tf.nn.bias_add(score,biases)
        return score



def get_xywh(box):

    x1,y1,x2,y2 = box
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = max(1., x2 - x1)
    h = max(1., y2 - y1)
    pos = [y,x]
    target_sz = [h,w]
    return pos,target_sz

def getSubWinTracking(img, pos, modelSz, originalSz, avgChans):

    #这个函数，获取用于填充并得到放入网络的模板
    if originalSz is None:
        originalSz = modelSz

    sz = originalSz
    im_sz = img.shape
    # make sure the size is not too small
    assert min(im_sz[:2]) > 2, "the size is too small"
    c = (np.array(sz) + 1) / 2

    # check out-of-bounds coordinates, and set them to black
    context_xmin = round(pos[1] - c[1])     #x1
    context_xmax = context_xmin + sz[1]  #y1
    context_ymin = round(pos[0] - c[0])     #x2
    context_ymax = context_ymin + sz[0] #y2

    #要填充的维度，
    left_pad = max(0, int(-context_xmin))
    top_pad = max(0, int(-context_ymin))
    right_pad = max(0, int(context_xmax - im_sz[1]))
    bottom_pad = max(0, int(context_ymax - im_sz[0]))

    context_xmin = int(context_xmin + left_pad)
    context_xmax = int(context_xmax + left_pad)
    context_ymin = int(context_ymin + top_pad)
    context_ymax = int(context_ymax + top_pad)


    #np.pad 其实和tf.pad一样， mode为填补类型（怎样去填补），如果为constant模式，就得指定填补的值，如果不指定，则默认填充0。
    # array:（上下），（左右）
    #三通道都填充,这里需要注意的是:r,b,g是二维数组
    if top_pad or left_pad or bottom_pad or right_pad:
        r = np.pad(img[:, :, 0], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                   constant_values=avgChans[0])
        g = np.pad(img[:, :, 1], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                   constant_values=avgChans[1])
        b = np.pad(img[:, :, 2], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                   constant_values=avgChans[2])


        r = np.expand_dims(r, 2)
        g = np.expand_dims(g, 2)
        b = np.expand_dims(b, 2)

        img = np.concatenate((r, g, b ), axis=2)

    im_patch_original = img[context_ymin:context_ymax, context_xmin:context_xmax, :]
    if not np.array_equal(modelSz, originalSz):
        im_patch = cv2.resize(im_patch_original, modelSz,interpolation=cv2.INTER_CUBIC)

    else:
        im_patch = im_patch_original
    print('1',im_patch.shape)

    return im_patch


def makeScalePyramid(im, targetPosition, in_side_scaled, out_side, avgChans):
    """
    先在target position周围 crop出一个search region
    先crop出一个max_target_side大小的patch，然后resize到beta * max
    然后呢 对着这个search region crop出多个尺度的patch 然后resize到out_side

    """
    in_side_scaled = np.round(in_side_scaled)
    max_target_side = int(round(in_side_scaled[-1]))
    min_target_side = int(round(in_side_scaled[0]))
    beta = out_side / float(min_target_side)

    search_side = int(round(beta * max_target_side))
    search_region = getSubWinTracking(im, targetPosition, (search_side, search_side),
                                              (max_target_side, max_target_side), avgChans)


    tmp_list = []
    tmp_pos = (search_side / 2., search_side / 2.)  #x,y
    for s in range(NUM_SCALES):
        target_side = round(beta * in_side_scaled[s])
        tmp_region = getSubWinTracking(search_region, tmp_pos, (out_side, out_side), (target_side, target_side),
                                               avgChans)
        tmp_list.append(tmp_region)

    pyramid = np.stack(tmp_list)

    return pyramid


def updata(score,area,pos,window):
    responseMaps = score[:, :, :, 0]
    _ares = 17 * 17
    responseMapsUP = []

    id = int(NUM_SCALES / 2)
    best_id = id
    bestPeak = -float('Inf')

    for i in range(NUM_SCALES):
        responseMapsUP.append(cv2.resize(responseMaps[i,:,:],(_ares,_ares),interpolation=cv2.INTER_CUBIC))
        thisResponse = responseMapsUP[-1]
        if i != id:
            thisResponse = thisResponse * 0.9745
        thisPeak = np.max(thisResponse)
        if thisPeak > bestPeak:
            bestPeak = thisPeak
            best_id = i
    responseMap = responseMapsUP[best_id]
    responseMap = responseMap - np.min(responseMap)
    responseMap = responseMap / np.sum(responseMap)
    responseMap = (1- 0.175) * responseMap + 0.175 * window
    rMax, cMax = np.unravel_index(responseMap.argmax(), responseMap.shape)
    pCorr = np.array((rMax, cMax))
    dispInstanceFinal = pCorr-int(_ares/2)
    dispInstanceInput = dispInstanceFinal*8/16
    dispInstanceFrame = dispInstanceInput*area/255
    new_pos = pos + dispInstanceFrame


    return new_pos,best_id






if __name__ == '__main__':



    siam = Siam_Network(False)
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, 'C:\\Users\\ZD\\Desktop\\tensorflow-siamese-fc-master\\model\\model_tf.ckpt')
    with tf.variable_scope('siamese') as scope:
        scope.reuse_variables()
        scores = siam.build_target_layer(siam.target_holder)   #模板区域  z
        # print(scores)

    video_path = 'C:\\Users\\ZD\\AppData\\Local\\Programs\\Python\\Python35\\car_person\\video_test2.mp4'
    video = cv2.VideoCapture(video_path)
    _, frame = video.read()
    bbox = cv2.selectROI(frame, False)
    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] #x1,y1,x2,y2
    pos, target_sz = get_xywh(bbox)
    avg = np.mean(frame,axis = (0,1))
    wcz = target_sz[1] + contextAmount * np.sum(target_sz)
    hcz = target_sz[0] + contextAmount * np.sum(target_sz)
    print(wcz,hcz)

    area = np.sqrt(wcz*hcz)
    scale = 127 / area
    target_crop = getSubWinTracking(frame,pos,(127,127),(np.around(area),np.around(area)),avg)

    pad =((255 - 127) / 2) / scale
    search_x = area + 2 * pad
    min_search_x = .2 * search_x
    max_search_x = 5. * search_x

    cos_window_size = 17 * 17
    cos_window = np.hanning(cos_window_size).reshape(cos_window_size,1)
    cos_window = cos_window.dot(cos_window.T)
    cos_window = cos_window / np.sum(cos_window)


    scales = np.array([scaleStep ** i for i in range(int(np.ceil(NUM_SCALES/2.0)-NUM_SCALES), int(np.floor(NUM_SCALES/2.0)+1))])
    print(scales)


    target_crop = np.expand_dims(target_crop,axis=0)

    zFeat = sess.run(scores,feed_dict={siam.target_holder:target_crop})
    zFeat = np.transpose(zFeat,[1,2,3,0])
    zFeat = tf.constant(zFeat,tf.float32)
    print(zFeat)
    scoreOp = siam.test_network(siam.search_holder,zFeat)
    # saver.restore(sess,'C:\\Users\\ZD\\Desktop\\tensorflow-siamese-fc-master\\model\\model_tf.ckpt')

    count_frame = 0
    while True:
        if count_frame == 0:
            count_frame += 1
            continue
        _, frame = video.read()
        if not _:
            break
        scale_search = search_x * scales

        scale_target = np.array([np.array(target_sz,np.float) * s for s in scales])
        x_crop = makeScalePyramid(frame,pos,scale_search,255,avg)
        # print(x_crop.shape)
        score = sess.run(scoreOp,feed_dict={siam.search_holder:x_crop})
        # print(score.shape)
        new_pos,new_scale = updata(score,round(area),pos,cos_window)
        print(scale_search)

        pos = new_pos
        area = max(min_search_x,min(max_search_x,(1 - .6) * area + .6 * scale_search[new_scale]))


        target_sz = (1 - .6) * np.array(target_sz,dtype=np.float) + .6 * scale_target[new_scale]

        rectPosition =  pos - target_sz / 2.
        p1 = tuple(np.round(rectPosition).astype(int)[::-1])
        p2 = tuple(np.round(rectPosition + target_sz).astype(int)[::-1])


        cv2.rectangle(frame, p1, p2, (255, 255, 255), thickness=3)
        cv2.imshow('hello',frame)
        count_frame += 1
        if cv2.waitKey(1) & 0xff == 27:
            break