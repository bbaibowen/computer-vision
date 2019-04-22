import numpy as np

feature_shape = [38,19,10,5,3,1]
scales = np.array([0.1,0.26,0.42,0.58,0.74,0.9])
ratios = [
    [0.5, 1.0, 2.0, 3.0,1/3.0],
    [0.5, 1.0, 2.0, 3.0, 1 / 3.0],
    [0.5, 1.0, 2.0, 3.0, 1 / 3.0],
    [0.5, 1.0, 2.0, 3.0, 1 / 3.0],
    [0.5, 1.0, 2.0, 3.0, 1 / 3.0],
    [0.5, 1.0, 2.0, 3.0, 1 / 3.0]
]
feature_maps_shape = [[None, 38, 38, 150], [None, 19, 19, 150], [None, 10, 10, 150], [None, 5, 5, 150], [None, 3, 3, 150], [None, 1, 1, 150]]

#
def anchors():
    all_anchors = []
    for i,j in enumerate(feature_shape):
        w = h = int(j)
        scale = scales[i]
        for x in range(w):
            for y in range(h):
                for ratio in ratios[i]:
                    x_cen = (x / float(w)) + (0.5 / float(w))
                    y_cen = (y / float(h)) + (0.5 / float(h))
                    width = scale * np.sqrt(ratio) / 1.2
                    height = scale / np.sqrt(ratio) / 1.2
                    all_anchors.append([x_cen,y_cen,width,height])
                all_anchors.append([(x / float(w)) + (0.5 / float(w)),(y / float(h)) + (0.5 / float(h)),scale * 1.5,scale * 1.4])
    all_anchors = np.array(all_anchors)
    return all_anchors

ANCHORS = anchors()

if __name__ == '__main__':
    a = anchors()
    print(a.shape)


