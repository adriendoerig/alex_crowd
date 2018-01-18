import numpy as np
from scipy import ndimage, misc
import random
import os

# input patches should be n_patches x 227 x 227 x 3 and labels should be a n_patches vector of 0s and 1s
def make_dataset_from_patch(patch_folder, image_size=(227,227,3), resize_factor=1.0, n_repeats=10, print_shapes=False):

    patch_folder = patch_folder+'/'

    min_num_images = 1
    num_images = 0

    image_files = os.listdir(patch_folder)

    data = np.ndarray(shape=(n_repeats * len(image_files), image_size[0], image_size[1], 3),
                           dtype=np.float32)
    labels = np.zeros(n_repeats * len(image_files), dtype=np.float32)

    print('loading images from ' + patch_folder)

    for image in image_files:

        image_file = os.path.join(patch_folder, image)

        for rep in range(n_repeats):
            try:

                this_image = ndimage.imread(image_file, mode='RGB').astype(float)
                this_image = misc.imresize(this_image, resize_factor)

                # crop out a random patch if the image is larger than image_size
                if this_image.shape[0] > image_size[0]:
                    firstRow = int((this_image.shape[0] - image_size[0]) / 2)
                    this_image = this_image[firstRow:firstRow + image_size[0], :, :]
                if this_image.shape[1] > image_size[1]:
                    firstCol = int((this_image.shape[1] - image_size[1]) / 2)
                    this_image = this_image[:, firstCol:firstCol + image_size[1], :]

                # pad to the right size if image is small than image_size (image will be in a random place)
                if any(np.less(this_image.shape,image_size)):
                    posX = random.randint(0,max(0,image_size[1]  - this_image.shape[1]))
                    posY = random.randint(0, max(0, image_size[0] - this_image.shape[0]))
                    padded = np.zeros(image_size, dtype=np.float32)
                    padded[posY:posY+this_image.shape[0], posX:posX+this_image.shape[1], :] = this_image
                    this_image = padded

                # normalize etc.
                # zero mean, 1 stdev
                this_image = (this_image - np.mean(this_image)) / np.std(this_image)

                # add to dataset
                data[num_images, :, :, :] = this_image

                if 'L' in image_file:
                    labels[num_images] = 0
                elif 'R' in image_file:
                    labels[num_images] = 1
                else:
                    raise Exception(image_file + ' is a stimulus of unknown class')

                num_images = num_images + 1

            except ():
                print('Could make image from patch:', image_file, '- it\'s ok, skipping.')

    # check enough images could be processed
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    # remove empty entries
    data = data[0:num_images, :, :, :]

    perm = np.random.permutation(num_images)
    data = data[perm, :, :, :]
    labels = labels[perm]

    if print_shapes:
        print('Dataset tensor:', data.shape)
        print('Mean:', np.mean(data))
        print('Standard deviation:', np.std(data))
    
    return data, labels
