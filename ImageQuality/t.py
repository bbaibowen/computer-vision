import cv2
from algorithm import ImageQuality,NR_Blur

im = cv2.imread('/home/baibowen/faceQuality/testImage/index10.jpg')
test= ImageQuality()
# test("NR_Blur",im)
# test.getResult()
# print(NR_Blur(im))
test.run_directories(image_dir = '/home/baibowen/faceQuality/testImage',csv_output = "./mycsv.csv",
                                      args = [{"method":"Brenner"},
                                              {"method":"Tenengrad"},
                                              {"method":"Laplacian"},
                                              {"method":"SMD"},
                                              {"method":"SMD","v2":True},
                                              {"method":"Energy"},
                                              # {"method":"EAV"},
                                              {"method":"Entropy"},
                                              {"method":"Vollath"},
                                              {"method":"NR_Blur"},
                                              {"method":"Dom"},
                                              {"method":"FFT"}])
# #more