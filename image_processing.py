import cv2, os
import numpy as np
import math


IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)


class Image:
    def __init__(self, img=None, steer=0):
        self.img = img
        self.steer = steer

    # Load image in BGR mode
    def load(self, data_dir, img_file):
        self.img = cv2.imread(os.path.join(data_dir, img_file.strip()))

    # Remove part of sky and car
    def crop(self, top, bottom):
        self.img = self.img[top:-bottom, :, :]

    # Resize to one shape
    def resize(self, width, height):
        self.img = cv2.resize(self.img, (width, height), cv2.INTER_AREA)

    # From NVIDIA model
    def to_yuv(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)

    # Simulate car in opposite situation
    def rand_flip(self):
        if np.random.choice(2):
            self.img = cv2.flip(self.img, 1)
            self.steer = -self.steer

    # Translate image horizontally and change steering (simulate different position on road)
    # and vertically (simulate hills)
    def translate_image(self, trans_x_range = 100, trans_y_range = 40):
        trans_x = trans_x_range * np.random.rand() - trans_x_range / 2
        self.steer = self.steer + trans_x * 0.004
        trans_y = trans_y_range * np.random.rand() - trans_y_range / 2
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = self.img.shape[:2]
        self.img = cv2.warpAffine(self.img, trans_m, (width, height))

    # Add random shadow on the screen, change brightness partially
    def add_shadow(self):
        # horizontal or vertical
        height, width = self.img.shape[:2]
        if np.random.choice(2):
            x1, y1 = width * np.random.rand(), 0
            x2, y2 = width * np.random.rand(), height
        else:
            x1, y1 = 0, height * np.random.rand()
            x2, y2 = width, height * np.random.rand()

        xm, ym = np.mgrid[0:height, 0:width]
        mask = np.zeros_like(self.img[:, :, 1])
        mask[np.where((ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0)] = 1

        cond = mask == np.random.randint(2)
        saturation = np.random.uniform(low=0.2, high=0.5)

        hls = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)
        hls[:, :, 1][cond] = hls[:, :, 1][cond] * saturation
        self.img = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)

    # Change image brightness randomly
    def brightness_augmentation(self, factor=0.5):
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        ratio = (factor + np.random.rand())
        hsv[:, :, 2] = hsv[:, :, 2] * ratio
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
        self.img = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2BGR)


# Choose random from 3 pictures, change steering to left and right picture (because of perspective).
def choose(data_dir, center, left, right, steering_angle):
    choice = np.random.choice(3)
    img = Image()
    if choice == 0:
        img.load(data_dir, left)
        return img, steering_angle + 0.25
    elif choice == 1:
        img.load(data_dir, center)
        return img, steering_angle
    elif choice == 2:
        img.load(data_dir, right)
        return img, steering_angle - 0.25


################################
# NVIDIA model
################################


def preprocessing_NVIDIA(img):
    img.crop(60, 25)
    img.resize(IMG_WIDTH, IMG_HEIGHT)
    img.to_yuv()

    return img


def augmentation(data_dir, center, left, right, steering_angle, range_x=100, range_y=40):
    img, angle = choose(data_dir, center, left, right, steering_angle)
    img.steer = angle
    img.rand_flip()
    img.translate_image(range_x, range_y)
    img.add_shadow()
    img.brightness_augmentation()
    return img


def batch_NVIDIA_gen(data_dir, img_paths, steering_angles, batch_size, train_cond):
    imgs = np.empty([batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(img_paths.shape[0]):
            center, left, right = img_paths[index]
            steering_angle = steering_angles[index]

            if train_cond and np.random.rand() < 0.6:
                m_img = augmentation(data_dir, center, left, right, steering_angle)
            else:
                m_img = Image()
                m_img.load(data_dir, center)

            imgs[i] = preprocessing_NVIDIA(m_img).img
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield imgs, steers


#####################################
# second model
#####################################


new_width, new_height = 64, 64
second_shape = (new_width, new_height, IMG_CHANNELS)

def preprocessing(img):
    img.crop(math.floor(img.img.shape[0]/5), 25)
    img.resize(new_width, new_height)

    return img

def batch_gen(data_dir, img_paths, steering_angles, batch_size, train_cond):
    imgs = np.empty([batch_size, new_height, new_width, IMG_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(img_paths.shape[0]):
            center, left, right = img_paths[index]
            steering_angle = steering_angles[index]

            if train_cond and np.random.rand() < 0.6:
                m_img = augmentation(data_dir, center, left, right, steering_angle)
            else:
                m_img = Image()
                m_img.load(data_dir, center)

            imgs[i] = preprocessing(m_img).img
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield imgs, steers
