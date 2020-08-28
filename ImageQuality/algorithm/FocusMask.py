from .include import *


def blur_mask(img):
    assert isinstance(img, np.ndarray), 'img_col must be a numpy array'
    assert img.ndim == 3, 'img_col must be a color image ({0} dimensions currently)'.format(img.ndim)
    msk, val, blurry = blur_detector(img)
    msk = cv2.convertScaleAbs(255-(255*msk/np.max(msk)))
    msk[msk < 50] = 0
    msk[msk > 127] = 255
    msk = remove_border(msk)
    msk = morphology(msk)
    result = np.sum(msk)/(255.0*msk.size)
    return result

def evaluate(img_col, thresh = 10):
    np.seterr(all='ignore')
    assert isinstance(img_col, np.ndarray), 'img_col must be a numpy array'
    assert img_col.ndim == 3, 'img_col must be a color image ({0} dimensions currently)'.format(img_col.ndim)
    img_gry = cv2.cvtColor(img_col, cv2.COLOR_RGB2GRAY)
    rows, cols = img_gry.shape
    crow, ccol = int(rows/2), int(cols/2)
    f = np.fft.fft2(img_gry)
    fshift = np.fft.fftshift(f)
    fshift[crow-75:crow+75, ccol-75:ccol+75] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_fft = np.fft.ifft2(f_ishift)
    img_fft = 20*np.log(np.abs(img_fft))

    result = np.mean(img_fft)
    return img_fft, result, result < thresh

def remove_border(msk, width=50):
    assert isinstance(msk, np.ndarray), 'msk must be a numpy array'
    assert msk.ndim == 2, 'msk must be a greyscale image'
    dh, dw = map(lambda i: i//width, msk.shape)
    h, w = msk.shape
    msk[:dh, :] = 255
    msk[h-dh:, :] = 255
    msk[:, :dw] = 255
    msk[:, w-dw:] = 255
    return msk

def morphology(msk):
    assert isinstance(msk, np.ndarray), 'msk must be a numpy array'
    assert msk.ndim == 2, 'msk must be a greyscale image'
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    msk = cv2.erode(msk, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, kernel)
    msk[msk < 128] = 0
    msk[msk > 127] = 255
    return msk

def blur_detector(img_col, thresh=10):
    assert isinstance(img_col, np.ndarray), 'img_col must be a numpy array'
    assert img_col.ndim == 3, 'img_col must be a color image ({0} dimensions currently)'.format(img_col.ndim)
    return evaluate(img_col=img_col, thresh=thresh)