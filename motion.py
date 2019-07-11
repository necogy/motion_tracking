import numpy as np
import cv2
import os
import collections
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

os.chdir("/Users/nzk180824h/xuan/fish_motion")
# the one with little glare
# cap = cv2.VideoCapture('85577.mp4')
# too much glare
# '85582.mp4')
# cap = cv2.VideoCapture('85551.mp4')
# cap = cv2.VideoCapture('85323.mp4')

video_list = ['85323.mp4','85555.mp4', '85582.mp4',	'85604.mp4', \
'85393.mp4', '85577.mp4','85600.mp4',\
'85551.mp4','85578.mp4','85602.mp4']

def l1(a, b, c, d):
    return np.sqrt((a - c) ** 2 + (b - d) ** 2)


def glare(color_img):
    frame_gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    # frame_gray = cv2.equalizeHist(frame_gray)
    frame_gray = cv2.medianBlur(frame_gray, 5)
    ret, th1 = cv2.threshold(frame_gray, 150, 255, cv2.THRESH_BINARY)
    kernel = np.ones((20, 20), np.uint8)
    mask = cv2.dilate(th1, kernel)
    dst = cv2.inpaint(color_img, mask, 3, cv2.INPAINT_NS)
    return dst


def too_much_glare(colored_img):
    # new_image = np.zeros(colored_img.shape, colored_img.dtype)
    # alpha, beta = 2.3, 0  # Simple contrast control
    # for y in range(colored_img.shape[0]):
    #     for x in range(colored_img.shape[1]):
    #         for c in range(colored_img.shape[2]):
    #             new_image[y, x, c] = np.clip(alpha * colored_img[y, x, c] + beta, 0, 255)
    channels = cv2.split(colored_img)
    eq_channels = []
    for ch, color in zip(channels, ['B', 'G', 'R']):
        eq_channels.append(cv2.equalizeHist(ch))

    eq_image = cv2.merge(eq_channels)
    eq_image = cv2.cvtColor(eq_image, cv2.COLOR_BGR2RGB)
    return eq_image


def draw_image_histogram(image, channels, color='k'):
    hist = cv2.calcHist([image], channels, None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])


def show_grayscale_histogram(image):
    draw_image_histogram(image, [0])
    plt.show()


def show_color_histogram(image):
    for i, col in enumerate(['b', 'g', 'r']):
        draw_image_histogram(image, [i], color=col)
    plt.show()


def bimodal(img):
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_img = cv2.threshold(grayscale_image, 150, 255, cv2.THRESH_TRUNC)
    return new_img


def contrast_checking(colored_img):
    frame_gray = cv2.cvtColor(colored_img, cv2.COLOR_BGR2GRAY)
    ratio = len(frame_gray[frame_gray > 150]) / len(frame_gray[frame_gray <= 150])
    counter = collections.Counter(frame_gray.ravel())
    peaks, properties = find_peaks(frame_gray.ravel(), distance=25000)
    return ratio

class video_processing:
    def __init__(self, video):
        self.video = video
        self.cap = cv2.VideoCapture(self.video)
        ret, self.first_frame = self.cap.read()

    def processing(self):
        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners = 100,
                               qualityLevel = 0.3,
                               minDistance = 7,
                               blockSize = 7 )
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15, 15),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        color = np.random.randint(0, 255, (100, 3))
        # Take first frame and find corners in it

        if contrast_checking(self.first_frame) <0.15:
            adjusted_frame = glare(self.first_frame)
            old_gray = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2GRAY)
        else:
            # adding part for adjusting the glare
            # adjusted_twice = too_much_glare(first_frame)
            frame_gray = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2GRAY)
            old_gray = cv2.equalizeHist(frame_gray)

        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        # Create a mask image for drawing purposes
        mask = np.zeros_like(self.first_frame)
        counter, sum_dis = 0, 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            # if adjusted == True:
            #     frame = glare(too_much_glare(frame))
            # else:
            #     frame = glare(frame)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.equalizeHist(frame_gray)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            if p1 is None:
                avg_motion_distance = sum_dis / counter
                print(avg_motion_distance)
                return avg_motion_distance
            good_new = p1[st==1]
            good_old = p0[st==1]
            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                counter += 1
                a, b = new.ravel()
                c, d = old.ravel()
                sum_dis += l1(a, b, c, d)
                mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
            img = cv2.add(frame, mask)
            cv2.imshow('frame', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        for i in range(10):
             cv2.destroyAllWindows()
             cv2.waitKey(1)

        self.cap.release()
        cv2.destroyAllWindows()

        # self.cap.release()


avg_motion_dis= video_processing(video_list[1]).processing()