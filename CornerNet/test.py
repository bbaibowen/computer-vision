import torch
import torch.nn as nn
import cv2
import numpy as np
from network import Model
from utils import decode_layer,soft_nms,nms

MODEL_PATH = 'CornerNet_500000.pkl'
model = Model()
img_path = ['../3.jpg']


class dect(object):

    def __init__(self,img_path = img_path,net = model,model_path = MODEL_PATH):

        self.img_path = img_path
        self.model = nn.DataParallel(net.eval())
        self.load_weights(model_path)
        self._mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self.img_size = 511


    def load_weights(self,model_path):

        self.model.load_state_dict(torch.load(model_path,'cpu'))

    def pull_img(self,img_path,transform = False):

        img = cv2.imread(img_path)
        # img = cv2.resize(img, (self.img_size, self.img_size))
        if transform:
            return self.transform(img)
        return img

    def transform(self,im):

        im = im / 255.
        im -= self._mean
        im /= self._std
        im = np.transpose(im,(2,0,1))
        im = np.expand_dims(im,axis=0)
        return im

    def crop_image(self,image, center, size):
        cty, ctx = center
        height, width = size
        im_height, im_width = image.shape[0:2]
        cropped_image = np.zeros((height, width, 3), dtype=image.dtype)

        x0, x1 = max(0, ctx - width // 2), min(ctx + width // 2, im_width)
        y0, y1 = max(0, cty - height // 2), min(cty + height // 2, im_height)

        left, right = ctx - x0, x1 - ctx
        top, bottom = cty - y0, y1 - cty

        cropped_cty, cropped_ctx = height // 2, width // 2
        y_slice = slice(cropped_cty - top, cropped_cty + bottom)
        x_slice = slice(cropped_ctx - left, cropped_ctx + right)
        cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

        border = np.array([
            cropped_cty - top,
            cropped_cty + bottom,
            cropped_ctx - left,
            cropped_ctx + right
        ], dtype=np.float32)

        offset = np.array([
            cty - height // 2,
            ctx - width // 2
        ])

        return cropped_image, border, offset

    def get_decode_out(self,img):
        outs = self.model(img)
        outs = decode_layer(outs,num_dets = 80)
        return outs.data.cpu().numpy()

    def draw(self,outs,size = None,path = None):
        if path is not None:
            im = cv2.imread(path)
            if size is not None:
                im = cv2.resize(im,size)
            for box in outs:
                p1 = (int(box[0]),int(box[1]))
                p2 = (int(box[2]),int(box[3]))
                cv2.rectangle(im, p1, p2, (255, 0, 0), 2)
            cv2.imshow('1',im)
            cv2.waitKey(0)
        else:
            for ind,path in enumerate(self.img_path):
                im = cv2.imread(path)
                if size is not None:
                    im = cv2.resize(im,size)
                for box in outs:
                    p1 = (int(box[0]),int(box[1]))
                    p2 = (int(box[2]),int(box[3]))
                    cv2.rectangle(im, p1, p2, (255, 0, 0), 1)
                cv2.imshow('1',im)
                cv2.waitKey(0)

    def to_xywh(self,box):

        x1,y2,x2,y1 = box[:,0],box[:,1],box[:,2],box[:,3]
        boxes = np.zeros_like(box,np.float32)
        boxes[:,0] = (x1 + x2) / 2
        boxes[:,1] = (y1 + y2) / 2
        boxes[:,3] = y2 - y1 + 1
        boxes[:,2] = x2 - x1 + 1

        return boxes



    def test_again(self,path):

        detections = []
        im = cv2.imread(path)
        height,width = im.shape[0:2]
        for scale in [1]:
            new_height = int(height * scale)
            new_width = int(width * scale)
            new_center = np.array([new_height // 2, new_width // 2])
            inp_height = new_height | 127
            inp_width  = new_width  | 127

            images = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
            ratios = np.zeros((1, 2), dtype=np.float32)
            borders = np.zeros((1, 4), dtype=np.float32)
            sizes = np.zeros((1, 2), dtype=np.float32)

            out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
            height_ratio = out_height / inp_height
            width_ratio = out_width / inp_width

            resized_image = cv2.resize(im, (new_width, new_height))
            resized_image, border, offset = self.crop_image(resized_image, new_center, [inp_height, inp_width])
            images[0]  = resized_image.transpose((2, 0, 1))
            borders[0] = border
            sizes[0]   = [int(height * scale), int(width * scale)]
            ratios[0]  = [height_ratio, width_ratio]

            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
            images = torch.from_numpy(images)
            dets = self.get_decode_out(images)

            dets = dets.reshape(2, -1, 8)
            dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]
            dets = dets.reshape(1, -1, 8)
            _rescale_dets(dets, ratios, borders, sizes)
            dets[:, :, 0:4] /= scale
            detections.append(dets)
        detections = np.concatenate(detections, axis=1)

        detections = detections[0]
        classes = detections[...,-1]
        boxes = []

        for i in range(1,81):

            index = (classes == i)
            box = np.array(detections[index][:,0:5],np.float)
            scores_index = (box[:,-1] > 0.4)
            box = box[scores_index]

            if box.shape[0] != 0:
                print('nmsj',box.shape)
                keep = nms(box,0.5)
                box = box[keep,:]
                print('nmsh',box.shape)
                boxes.append(box)
        boxes = np.concatenate(boxes,axis = 0)
        print(boxes.shape)


        self.draw(boxes,path=path)

def _rescale_dets(detections, ratios, borders, sizes):
    xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]
    xs    /= ratios[:, 1][:, None, None]
    ys    /= ratios[:, 0][:, None, None]
    xs    -= borders[:, 2][:, None, None]
    ys    -= borders[:, 0][:, None, None]
    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)


if __name__ == '__main__':
    t = dect()

    t.test_again('../3.jpg')



