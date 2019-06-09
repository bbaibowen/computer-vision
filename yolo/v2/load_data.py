import numpy as np
import h5py
import io
import PIL.Image as P


class load_data(object):
    def __init__(self,batch,path):
        self.batch = batch
        self.h5file = h5py.File(path,'r')
        self.start = 0

    def handle(self,img,box):
        im = P.open(io.BytesIO(img))
        size = np.array([im.width, im.height])
        size = np.expand_dims(size, axis=0)
        image = im.resize((416, 416), P.BICUBIC)
        image = np.array(image, np.float)
        image /= 255.
        image = np.expand_dims(image,axis=0)
        boxes = box.reshape((-1, 5))
        box_xy = .5 * (boxes[:, 3:5] + boxes[:, 1:3])
        box_wh = boxes[:, 3:5] - boxes[:, 1:3]
        box_xy = box_xy / size
        box_wh = box_wh / size
        boxes = np.concatenate((box_xy, box_wh, boxes[:, 0:1]), axis=1)
        # boxes = np.expand_dims(boxes,axis=0)

        return image,boxes




    def get(self):
        images = self.h5file['train/images'].value[self.start:self.start + self.batch]
        boxes = self.h5file['train/boxes'].value[self.start:self.start + self.batch]

        for i in range(self.batch):
            im,box = self.handle(images[i],boxes[i])
        self.start += self.batch
        return im,box


if __name__ == '__main__':
    lo = load_data(1,'voc2007.hdf5')
    for i in range(10):
        im,b = lo.get()
        print(im.shape,b.shape)