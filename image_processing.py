import cv2, os
import numpy as np
import matplotlib.image as mpimg

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

class image:
    def __init__(self, img = None, steer = 0):
        self.img = img
        self.steer = steer

    def load(self, data_dir, img_file):
        self.img = mpimg.imread(os.path.join(data_dir, img_file.strip()))

    def crop(self):
        self.img = self.img[60:-25, :, :]

    def resize(self):
        self.img = cv2.resize(self.img, (IMG_WIDTH, IMG_HEIGHT), cv2.INTER_AREA)

    def to_yuv(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2YUV)

    def rand_flip(self, steering_angle):
        if np.random.rand() < 0.5:
            self.img = cv2.flip(self.img, 1)
            self.steer = -steering_angle

    def rand_translate(self, range_x, range_y):
        trans_x = range_x * (np.random.rand() - 0.5)
        trans_y = range_y * (np.random.rand() - 0.5)
        self.steer += trans_x * 0.002
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = self.img.shape[:2]
        self.img = cv2.warpAffine(self.img, trans_m, (width, height))

    def rand_shadow(self):
        x1, y1 = IMG_WIDTH * np.random.rand(), 0
        x2, y2 = IMG_WIDTH * np.random.rand(), IMG_HEIGHT
        xm, ym = np.mgrid[0:IMG_HEIGHT, 0:IMG_WIDTH]
        mask = np.zeros_like(self.img[:, :, 1])
        mask[np.where((ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0)] = 1

        cond = mask == np.random.randint(2)
        saturation = np.random.uniform(low = 0.2, high = 0.5)

        hls = cv2.cvtColor(self.img, cv2.COLOR_RGB2HLS)
        hls[:, :, 1][cond] = hls[:, :, 1][cond] * saturation
        self.img = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

    def rand_brightness(self):
        hsv = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
        hsv[:, :, 2] = hsv[:, :, 2] * ratio
        self.img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def preproc(img):
    img.crop()
    img.resize()
    img.to_yuv()
    return img

def aug(img, data_dir, center, left, right, steering_angle, range_x = 100, range_y = 10):
    img, angle = choose(data_dir, center, left, right, steering_angle)
    img.steer = angle
    img.rand_flip(steering_angle)
    img.rand_translate(range_x, range_y)
    img.rand_shadow()
    img.rand_brightness()
    return img

def choose(data_dir, center, left, right, steering_angle):
    choice = np.random.choice(3)
    img = image()
    if choice == 0:
        img.load(data_dir, left)
        return img, steering_angle + 0.2
    elif choice == 1:
        img.load(data_dir, center)
        return img, steering_angle
    elif choice == 2:
        img.load(data_dir, right)
        return img, steering_angle - 0.2

def batch_gen(data_dir, img_paths, steering_angles, batch_size, train_cond):
    imgs = np.empty([batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
    steers = np.empty(batch_size)
    m_img = image()
    while True:
        i = 0
        for index in np.random.permutation(img_paths.shape[0]):
            center, left, right = img_paths[index]
            steering_angle = steering_angles[index]

            if train_cond and np.random.rand() < 0.6:
            	m_img = aug(m_img, data_dir, center, left, right, steering_angle)
            else:
            	m_img.load(data_dir, center)

            imgs[i] = preproc(m_img).img
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
            	break
        yield imgs, steers
