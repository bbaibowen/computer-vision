from .base import *
import pandas as pd

class ImageQuality:

    def __init__(self):
        self.result = None
        self.method = None
        self.runTime = None


    def __call__(self,method,image,**kwargs):
        result,runTime = globals().get(method)(image,**kwargs)
        self.result = "{:.2f}".format(result)
        self.runTime = "{:.2f}".format(runTime)
        self.method = method
        self.getResult()


    def getResult(self):
        ss = "Methods:{}\nresult:{}\nrunTime:{} ms  \n".format(self.method,self.result,self.runTime)
        print(ss)
        return self.result,ss


    def run_directories(self,**kwargs):
        '''
        you can input like:
            image_dir detail_args ,such as :
                image_dir=./images   args=[{"method":Energy,...},{"method":"FFT"}]
        :return:
        '''

        image_dir = kwargs['image_dir']
        image_lists = os.listdir(image_dir)
        data = []
        columns = ['image','Method','result','runTime(:ms)']
        for img_path in image_lists:
            image = cv2.imread(os.path.join(image_dir,img_path))
            for method in kwargs['args']:
                print("image:{}".format(img_path))
                self.__call__(image=image,**method)
                data.append([img_path,self.method,self.result,self.runTime])
                print('----------------------------------------------------------------------------------------------')
        df = pd.DataFrame(data)
        df.columns = columns
        df.to_csv('./test.csv' if "csv_output" not in kwargs else kwargs["csv_output"],index=False)


