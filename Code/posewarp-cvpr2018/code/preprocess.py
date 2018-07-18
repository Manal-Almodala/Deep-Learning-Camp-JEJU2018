import scipy.io as sio
import matplotlib.pyplot as plt
import cv2
import numpy as np
import param
import os
import tf_pose

from skimage.io import imsave

SKIP = 1

vid_path = "D:/Proyectos/JEJU2018/Data/selected_data/good/Golf-Swing-Front/004/RF1-13206_70024.avi"

def vid_to_seq(vid_path, save_path=None):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cap = cv2.VideoCapture(vid_path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount // SKIP + 1, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    f = 0
    ret = True

    while (fc < frameCount and ret):
        r, b = cap.read()
        if fc % SKIP == 0:

            ret, buf[f] = r, b
            buf[f] = buf[f][:, :, ::-1] # Inverts channels from BGR to RGB

            if save_path:
                imsave(save_path + "/" + str(f+1) + ".png", np.array(buf[f], dtype=np.uint8))

            f += 1

        fc += 1

    cap.release()
    return buf[:-1]


def seq_to_inf(frames, save_path=None):

    estimator = tf_pose.get_estimator(model="mobilenet_thin")

    J =  param.get_general_params()['n_joints']

    skl = np.zeros([J, 2, len(frames)]) - 1
    bbx = np.zeros([len(frames), 4]) - 1

    for f, fr in enumerate(frames):

        points = estimator.inference(fr, resize_to_default=(432 > 0 and 368 > 0), upsample_size=4.0)[0].body_parts

        min_x, min_y = (+np.inf, +np.inf)
        max_x, max_y = (-np.inf, -np.inf)

        imw, imh = fr[:, :, 0].shape

        for key in points:
            if key < J:
                # Get the coordinates of the bodypart.
                x, y = ((points[key].x) * imh, (points[key].y) * imw)

                min_x = np.minimum(min_x, x)
                min_y = np.minimum(min_y, y)
                max_x = np.maximum(max_x, x)
                max_y = np.maximum(max_y, y)

                skl[key, 0, f] = x
                skl[key, 1, f] = y

                # plt.plot(skl[:, 0], skl[:, 1], "o", c="red", markersize=2)

        # # Plot bound box based on skeleton joints.
        # plt.plot([min_x, max_x, max_x, min_x, min_x],
        #          [min_y, min_y, max_y, max_y, min_y], "-", c="yellow")

        bbx[f, :] = [min_x, min_y, max_x - min_x, max_y - min_y]

    info = {"data" : {"X" : skl[:J], "bbox" : bbx }}

    if save_path:
        sio.savemat(save_path + ".mat" , info)

    return info

# Split vid_path in directory and name.
vid_dir  = '/'.join(vid_path.split('/')[:-3]) + "/"
vid_name = '_'.join(vid_path.split('/')[-3:-1])

data_dir = param.get_general_params()['data_dir']

# frms = vid_to_seq(vid_path=vid_path, save_path=data_dir + "/train/frames/" + vid_name + "/")
# info = seq_to_inf(frms, save_path=data_dir + "/train/info/" + vid_name)
img = [cv2.imread("D:/Proyectos/JEJU2018/Code/posewarp-cvpr2018/data/test/frames/Carlos/1.png")[:, :, ::-1],
       cv2.imread("D:/Proyectos/JEJU2018/Code/posewarp-cvpr2018/data/test/frames/Carlos/2.png")[:, :, ::-1]]

info = seq_to_inf(img, save_path=data_dir + "/test/info/Carlos")

info_name = 'D:/Proyectos/JEJU2018/Code/posewarp-cvpr2018/data/train/info/Golf-Swing-Front_003.mat'
info = sio.loadmat(info_name)

fr = 7

img = cv2.imread("D:/Proyectos/JEJU2018/Code/posewarp-cvpr2018/data/train/frames/Golf-Swing-Front_003/" + str(fr + 1) + ".png")[:, :, ::-1]

bbox = info['data']['bbox'][0][0][fr]
sklt = np.array([[info['data']['X'][0][0][j][0][fr], info['data']['X'][0][0][j][1][fr]] for j in range(14)]).T

print(sklt)

pos = np.zeros(2)

x = [bbox[0], bbox[0] + bbox[2],  bbox[0] + bbox[2], bbox[0], bbox[0]]
y = [bbox[1], bbox[1],  bbox[1] + bbox[3],  bbox[1]+ bbox[3], bbox[1]]

print(pos)
plt.plot(x ,y, "-", c="yellow")
plt.plot(sklt[0], sklt[1], "o", c="red", markersize=2)

plt.imshow(img)
plt.show()

print("stop")
