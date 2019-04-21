import numpy as np
import cv2
import os
import tensorflow as tf
class Reader:
    def __init__(self,root_dir):
        self.root_dir=root_dir
        self.interval=30
        self.cate_list={}
        self.cate_box={}
        self.img_num=0
        self.img_list=[]
        self.label_list=[]
        self.node=[]
        self.input_list=[]
        with open(os.path.join(self.root_dir,'list.txt')) as f:
            line=f.readline().strip('\n')
            while (line):
                #===========label===============
                with open(os.path.join(self.root_dir,line,'groundtruth.txt')) as f2:
                    line2=f2.readline().strip('\n')
                    boxes=[]
                    while (line2):
                        box=line2.split(',')
                        boxes.append([int(float(box[0])),int(float(box[1])),int(float(box[2])),int(float(box[3]))])
                        line2=f2.readline().strip('\n')
                    self.cate_box[line]=np.array(boxes)
                #===========label===============

                #===========img_list============
                img_list=[x for x in os.listdir(os.path.join(self.root_dir,line)) if 'JPEG' in x or 'jpg' in x]
                img_list.sort()
                img_path=[]
                for name in img_list:
                    img_path.append(os.path.join(self.root_dir,line,name))
                self.cate_list[line]=np.array(img_path)
                #===========img_list============

                #=============filter============
                index=[]
                for i in range(len(self.cate_list[line])):
                    if not np.all(self.cate_box[line][i]==[0,0,0,0]):
                        #print('exception '+line+' '+str(i))
                        index.append(i)
                self.cate_box[line]=self.cate_box[line][index]
                self.cate_list[line]=self.cate_list[line][index]
                self.img_num+=len(self.cate_list[line])
                #=============filter============
                line=f.readline().strip('\n')
        print(self.img_num)
        #====================transform-list==========================
        self.node.append(0)
        cates=list(self.cate_list.keys())
        for i,cate in enumerate(cates):
            for j in range(len(self.cate_list[cate])):
                self.img_list.append(self.cate_list[cate][j])
                self.label_list.append(list(self.cate_box[cate][j]))
            self.node.append(len(self.img_list))
        self.input_list=list(range(self.node[-1]))
        #====================transform-list==========================

        #===================input-producer===========================
        self.input_list=tf.convert_to_tensor(self.input_list)
        self.img_list=tf.convert_to_tensor(self.img_list)
        self.label_list=tf.convert_to_tensor(self.label_list)
        self.node=tf.convert_to_tensor(self.node)
        self.queue=tf.train.slice_input_producer([self.input_list],shuffle=True)
        self.template_p,self.template_label_p,self.detection_p,self.detection_label_p,self.offset,self.ratio,self.detection,self.detection_label,self.index_t,self.index_d=self.read_from_disk(self.queue)
        #===================input-producer===========================
    def read_from_disk(self,queue):
        index_t=queue[0]#tf.random_shuffle(self.input_list)[0]
        index_min=tf.reshape(tf.where(tf.less_equal(self.node,index_t)),[-1])
        node_min=self.node[index_min[-1]]
        node_max=self.node[index_min[-1]+1]
        interval_list=list(range(30,100))
        interval=tf.random_shuffle(interval_list)[0]
        index_d=[tf.cond(tf.greater(index_t-interval,node_min),lambda:index_t-interval,lambda:index_t+interval),tf.cond(tf.less(index_t+interval,node_max),lambda:index_t+interval,lambda:index_t-interval)]
        index_d=tf.random_shuffle(index_d)
        index_d=index_d[0]

        constant_t=tf.read_file(self.img_list[index_t])
        template=tf.image.decode_jpeg(constant_t, channels=3)
        template=template[:,:,::-1]
        constant_d=tf.read_file(self.img_list[index_d])
        detection=tf.image.decode_jpeg(constant_d, channels=3)
        detection=detection[:,:,::-1]

        template_label=self.label_list[index_t]
        detection_label=self.label_list[index_d]

        template_p,template_label_p,_,_=self.crop_resize(template,template_label,1)
        detection_p,detection_label_p,offset,ratio=self.crop_resize(detection,detection_label,2)

        return template_p,template_label_p,detection_p,detection_label_p,offset,ratio,detection,detection_label,index_t,index_d


    def crop_resize(self,img,label,rate=1,random_patch=True):
        #label=[x,y,w,h]===x,y is left-top corner
        x=label[0]
        y=label[1]
        w=label[2]
        h=label[3]
        img=tf.cast(img,tf.float32)
        mean_axis=tf.cast(tf.to_int32(tf.reduce_mean(img,axis=(0,1))),tf.float32)
        p=tf.to_int32((w+h)/2)
        s=(w+p)*(h+p)
        side=tf.to_int32(tf.round(tf.sqrt(tf.to_float(s))*rate))
        x1=tf.to_int32(x-tf.to_int32((side-w)/2))
        y1=tf.to_int32(y-tf.to_int32((side-h)/2))
        x2=tf.to_int32(x1+side)
        y2=tf.to_int32(y1+side)

        offset=[x1,y1]

        pad_l=tf.cond(tf.less(x1,0),lambda:tf.abs(x1),lambda:0)
        pad_r=tf.cond(tf.greater(x2,tf.shape(img)[1]),lambda:x2-tf.shape(img)[1]+1,lambda:0)
        pad_u=tf.cond(tf.less(y1,0),lambda:tf.abs(y1),lambda:0)
        pad_d=tf.cond(tf.greater(y2,tf.shape(img)[0]),lambda:y2-tf.shape(img)[0]+1,lambda:0)

        x=tf.cond(tf.less(x1,0),lambda:x-x1,lambda:x)
        y=tf.cond(tf.less(y1,0),lambda:y-y1,lambda:y)

        img_b,img_g,img_r=tf.unstack(img,3,axis=2)
        img_b=tf.pad(img_b,[[pad_u,pad_d],[pad_l,pad_r]],constant_values=mean_axis[0])
        img_g=tf.pad(img_g,[[pad_u,pad_d],[pad_l,pad_r]],constant_values=mean_axis[1])
        img_r=tf.pad(img_r,[[pad_u,pad_d],[pad_l,pad_r]],constant_values=mean_axis[2])
        img=tf.stack([img_b,img_g,img_r],axis=2)

        x1=tf.to_int32(x-tf.to_int32((side-w)/2))
        y1=tf.to_int32(y-tf.to_int32((side-h)/2))
        x2=tf.to_int32(x1+side)
        y2=tf.to_int32(y1+side)

        if rate==1:
            crop_img=img[y1:y2,x1:x2,:]
            resize_img=tf.image.resize_images(crop_img,(127,127))/255.
            ratio=tf.to_float(side)/127.
            label=tf.cast([63,63,tf.to_float(w)/ratio,tf.to_float(h)/ratio],tf.int32)
            label=tf.cast(label,tf.float32)
        if rate==2:
            random_x=0
            random_y=0
            if random_patch:
                out_b=tf.shape(img)[1]-x2
                in_b=x-x1
                shift_max_x=tf.to_float(tf.cond(tf.less_equal(out_b,in_b),lambda:out_b,lambda:in_b))
                out_b=x1
                in_b=x2-(x+w)
                shift_min_x=tf.to_float(-tf.cond(tf.less_equal(out_b,in_b),lambda:out_b,lambda:in_b))
                out_b=tf.shape(img)[0]-y2
                in_b=y-y1
                shift_max_y=tf.to_float(tf.cond(tf.less_equal(out_b,in_b),lambda:out_b,lambda:in_b))
                out_b=y1
                in_b=y2-(y+h)
                shift_min_y=tf.to_float(-tf.cond(tf.less_equal(out_b,in_b),lambda:out_b,lambda:in_b))

                random_x=tf.cast(tf.random_uniform([],shift_min_x,shift_max_x),tf.int32)
                random_y=tf.cast(tf.random_uniform([],shift_min_y,shift_max_y),tf.int32)

                x1=x1+random_x
                x2=x2+random_x
                y1=y1+random_y
                y2=y2+random_y

                offset[0]=offset[0]+random_x
                offset[1]=offset[1]+random_y

                crop_img=img[y1:y2,x1:x2,:]
            else:
                crop_img=img[y1:y2,x1:x2,:]
            resize_img=tf.image.resize_images(crop_img,(255,255))/255.
            ratio=tf.to_float(side)/255.
            label=tf.cast([tf.to_float(127-tf.to_float(random_x)/ratio),tf.to_float(127-tf.to_float(random_y)/ratio),tf.to_float(w)/ratio,tf.to_float(h)/ratio],tf.int32)
            label=tf.cast(label,tf.float32)

        return resize_img,label,offset,ratio
    def get_batch(self,batch_size=1):

        template_p,template_label_p,detection_p,detection_label_p,offset,ratio=tf.train.batch\
        ([self.template_p,self.template_label_p,self.detection_p,self.detection_label_p,self.offset,self.ratio],\
            batch_size,num_threads=32,capacity=2048,shapes=[(127,127,3),(4),(255,255,3),(4),(2),()])
        return template_p,template_label_p,detection_p,detection_label_p,offset,ratio

