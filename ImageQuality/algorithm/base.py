from .include import *
from .dom import DOM
from .FocusMask import blur_mask


@calculate_function_run_time_ms
def Brenner(image):
    '''
    method1:Brenner 梯度函数
        计算相邻两个像素灰度差的平方

    :param image:
    :return:
    '''
    assert image is not None
    #2gray
    gray_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h,w = gray_image.shape
    out = 0
    for y in range(w):
        for x in range(0, h - 2):
            out += (int(gray_image[x + 2, y]) - int(gray_image[x, y])) ** 2

    return out

@calculate_function_run_time_ms
def Tenengrad(image):
    '''
    Tenengrad 梯度函数,用Sobel算子分别提取水平和垂直方向的梯度值
    :param image:
    :return:
    '''
    assert image is not None
    gray_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    sobel_x = cv2.Sobel(gray_image,cv2.CV_32FC1,1,0)
    sobel_y = cv2.Sobel(gray_image,cv2.CV_32FC1,0,1)
    sobel_xx = cv2.multiply(sobel_x,sobel_x)
    sobel_yy = cv2.multiply(sobel_y,sobel_y)
    image_gradient = sobel_xx + sobel_yy
    image_gradient = np.sqrt(image_gradient).mean()
    return image_gradient

@calculate_function_run_time_ms
def Laplacian(image):
    '''
    Laplacian 梯度函数:Laplacian算子
    :param image:
    :return:
    '''
    assert image is not None
    gray_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    image_gradient = cv2.Laplacian(gray_image,cv2.CV_32FC1)
    image_gradient = np.abs(image_gradient).mean()
    return image_gradient

@calculate_function_run_time_ms
def SMD(image,v2 = False):
    '''
    灰度方差，将灰度变化作为聚焦评价的依据
    灰度方差乘积,对每一个像素领域两个灰度差相乘后再逐个像素累加

    :param image:
    :param 灰度方差乘积
    :return:
    '''
    assert image is not None
    gray_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    kerner_x = kerner_y = np.zeros((3,3),dtype=np.float32)
    kerner_x[1,2] = -1.;kerner_x[1,1] = 1.
    kerner_y[0,1] = -1.;kerner_y[1,1] = 1.
    smd_x = cv2.filter2D(gray_image,cv2.CV_32FC1,kerner_x)
    smd_y = cv2.filter2D(gray_image,cv2.CV_32FC1,kerner_y)
    result = (np.abs(smd_x) + np.abs(smd_y)).mean() if not v2 else np.multiply(np.abs(smd_x),np.abs(smd_y)).mean()
    return result

@calculate_function_run_time_ms
def Energy(image):
    '''
    能量梯度，
    :param image:
    :return:
    '''
    assert image is not None
    gray_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    kerner_x = kerner_y = np.zeros((3,3),dtype=np.float32)
    kerner_x[1, 2] = -1.;kerner_x[1, 1] = 1.
    kerner_y[2, 1] = -1.;kerner_y[1, 1] = 1.
    energy_x = cv2.filter2D(gray_image, cv2.CV_32FC1, kerner_x)
    energy_y = cv2.filter2D(gray_image, cv2.CV_32FC1, kerner_y)
    return (np.multiply(energy_x,energy_x) + np.multiply(energy_y,energy_y)).mean()

@calculate_function_run_time_ms
def EAV(image):
    '''
    点锐度函数,很慢
    :param image:
    :return:
    '''
    assert image is not None
    gray_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    result = 0
    for i in range(1,gray_image.shape[0]-1):
        for j in range(1,gray_image.shape[1]-1):
            cur = gray_image[i - 1];cur_mul = gray_image[i - 1, j - 1];cur_add = gray_image[i - 1, j + 1]
            prev = gray_image[i - 1];prev_mul = gray_image[i-1,j-1];prev_add = gray_image[i-1,j+1]
            next = gray_image[i + 1];next_mul = gray_image[i+1,j-1];next_add = gray_image[i+1,j+1]
            this = abs(prev_mul - cur) * 0.7 + abs(prev - cur) + abs(prev_add - cur) * 0.7 + \
                abs(next_mul - cur) * 0.7 + abs(next - cur) + abs(next_add - cur) * 0.7 + \
                abs(cur_mul - cur) + abs(cur_add - cur)
            result += this
    return result.mean()

@calculate_function_run_time_ms
def Entropy(image):
    '''
    :param img:narray
    :return: float
    模糊和清晰图像的清晰度的熵差别不大
    '''
    assert image is not None
    gray_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    out = 0
    count = gray_image.shape[0]*gray_image.shape[1]
    p = np.bincount(np.array(gray_image).flatten())
    for i in range(0, len(p)):
        if p[i]!=0:
            out-=p[i]*math.log(p[i]/count)/count
    return out

@calculate_function_run_time_ms
def Vollath(image):
    '''
    :param img:narray
    :return: float
    '''
    assert image is not None
    gray_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    shape = np.shape(gray_image)
    u = np.mean(gray_image)
    out = -shape[0]*shape[1]*(u**2)
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]):
            out+=int(gray_image[x,y])*int(gray_image[x+1,y])
    return out

#No-Reference Image Quality Assessment using Blur and Noise
@calculate_function_run_time_ms
def NR_Blur(image):
    '''
    只用《#No-Reference Image Quality Assessment using Blur and Noise》的blur部分
    :param image:
    :return:
    '''
    assert image is not None
    gray_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    #水平、竖直差值梯度
    kernel_h = kernel_v = np.zeros((3, 3), np.float32)
    kernel_h[0,1] = -1.;kernel_h[2,1] = 1.
    kernel_v[1,0] = -1.;kernel_v[1,2] = 1.
    grad_h = cv2.filter2D(gray_image, cv2.CV_32FC1, kernel_h)
    grad_v = cv2.filter2D(gray_image, cv2.CV_32FC1, kernel_v)

    #筛选边缘点 D_h > D_mean
    mean = np.mean(grad_v)
    mask = grad_h > mean
    mask = mask / 255
    c_h = grad_h * mask
    #//进一步筛选边缘点 C_h(x,y) > C_h(x,y-1) and C_h(x,y) > C_h(x,y+1)
    edge = np.zeros(c_h.shape[:2],np.uint8)
    for i in range(1,c_h.shape[0] - 1):
        for j in range(c_h.shape[1]):
            prev = c_h[i-1,j]
            cur = c_h[i,j]
            next = c_h[i+1,j]
            if (prev < cur and next < cur):
                edge[i,j] = 1

    #检测边缘点是否模糊
    a_h = grad_h / 2
    gray_image = gray_image.astype(np.float32)
    br_h = cv2.absdiff(gray_image,a_h)
    br_h /= a_h

    a_v = grad_v / 2
    br_v = cv2.absdiff(gray_image,a_v)
    br_v /= a_v

    inv_blur = np.zeros(br_v.shape,dtype=np.float32)
    for i in range(inv_blur.shape[0]):
        for j in range(inv_blur.shape[1]):
            data_v = br_v[i,j]
            data_h = br_h[i,j]
            inv_blur[i,j] = data_v if data_v > data_h else data_h
    blur = (inv_blur < 0.1) / 255

    #计算边缘模糊的均值和比例
    sum_inv_blur = cv2.countNonZero(inv_blur)
    sum_blur = cv2.countNonZero(blur)
    sum_edge = cv2.countNonZero(edge)
    blur_mean = sum_inv_blur / sum_blur
    blur_ratio = sum_blur / sum_edge
    return blur_ratio

@calculate_function_run_time_ms
def Dom(image,**kwargs):
    return DOM().get_sharpness(image,**kwargs)

@calculate_function_run_time_ms
def FFT(image):
    return blur_mask(image)
