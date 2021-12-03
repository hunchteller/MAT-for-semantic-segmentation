import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets o.  The jaccard overlap
    is simply the intersection over union of tw.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple boundin, Shape: [nu,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, labels):
        for t in self.transforms:
            img, labels = t(img, labels)
        return img, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, labels):
        return self.lambd(img, labels)


class ConvertFromInts(object):
    def __call__(self, image=None, labels=None):
        return image.astype(np.float32), labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image=None, labels=None):
        image = cv2.resize(image, (self.size,
                                   self.size))
        return image, labels

class Rotate(object):
    def __init__(self, deg):
        self.deg = deg

    def __call__(self, image, label):
        deg = random.randint(-self.deg, self.deg)
        image = cv2.rotate(image, deg)

class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
        return image, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, labels


class ToCV2Image(object):
    def __call__(self, tensor=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), labels


class ToTensor(object):
    def __call__(self, cvimage=None, labels=None, label2=None):
        if label2:
            return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1).contiguous(), torch.from_numpy(
                labels).long(), torch.from_numpy(label2).long(),
        else:
            return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1).contiguous(), torch.from_numpy(
                labels).long()


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, labels):
        if random.randint(2):
            return image, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
        int(left):int(left + width)] = image
        image = expand_image
        return image, labels


class RandomMirror(object):
    """
    index: mirror direction, 1 horizontal, 0 vertical
    """
    def __init__(self, index=1):
        self.index = index

    def __call__(self, image, lbl):
        _, width, _ = image.shape
        if random.randint(2):
            image = cv2.flip(image, self.index)
            lbl = cv2.flip(lbl, self.index)
        return image, lbl


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, lbl):
        h, w, c = img.shape
        if h > 300:
            h0 = random.randint(0, h - self.size)
            w0 = random.randint(0, w - self.size)
            img = img[h0:h0 + self.size, w0:w0 + self.size]
            lbl = lbl[h0:h0 + self.size, w0:w0 + self.size]
        return img, lbl


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, labels):
        im = image.copy()
        im, labels = self.rand_brightness(im, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, labels = distort(im, labels)
        return self.rand_light_noise(im, labels)

class Resize(object):
    def __init__(self, size, train=True):
        self.sizes = size
        self.train = train

    def __call__(self, img, label):
        i = random.randint(0, len(self.sizes))
        h = self.sizes[i]
        img = cv2.resize(img, (h, h), cv2.INTER_LINEAR)
        if self.train:
            label = cv2.resize(label, (h, h), cv2.INTER_NEAREST)
        return img, label

class trainAugmentation(object):
    def __init__(self, size=300):
        self.size = size

        self.augment = Compose([
            # Resize([300, 400, 500, 512, 600, 800]),
            RandomCrop(300),
            # RandomMirror(),
            # RandomMirror(0),
            # PhotometricDistort(),
            ToTensor()
        ])
        self.toTensor = ToTensor()

    def __call__(self, img, labels):
        return self.augment(img, labels)


class valAugmentation(object):
    def __init__(self, size):
        self.augment = Compose([
            # Resize(size, train=False),
            ToTensor(),
        ])

    def __call__(self, img, labels):
        return self.augment(img, labels)
