import os
import skimage.io as io
import skimage.transform as transform
import cv2
import numpy as np


# Coloca uma linha verde nos limites da máscara
def limit(mask, img):
    for i in range(255):
        for j in range(255):
            if ((not (mask[i, j] == mask[i, j + 1] == mask[i, j - 1] == mask[i - 1, j] ==
                      mask[i - 1, j - 1] == mask[i - 1, j + 1] == mask[i + 1, j] ==
                      mask[i + 1, j - 1] == mask[i + 1, j + 1])) and (mask[i, j] != 0)):
                    img[i, j, 1:2] = 1  # colocar uma linha verde nos 

    return img


def adjust_mask(mask):
    mean = np.mean(mask)
    mask[mask > mean] = 1
    mask[mask <= mean] = 0

    return mask[0]


def normalize_pixel_values(img):

    img = img - np.mean(img)
    img = img / np.std(img)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    # Transformação linear que coloca os valores entre 0 e 1
    return img


def crop_mask(img_dir, mask_dir, template_dir):

    img = cv2.imread(img_dir, 0)
    mask = cv2.imread(mask_dir, 0)
    img2 = img.copy()
    template = cv2.imread(template_dir, 0)
    h, w = template.shape

    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    method = cv2.TM_SQDIFF_NORMED
    img = img2.copy()

    # Apply template Matching
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cropped_mask = mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    return cropped_mask


def resize_and_save(save_img_path, save_mask_path):

    img_path = "D:/ISIC_Skin_Cancer/ISIC-2017_Validation_Data/all/"
    mask_path = "D:/ISIC_Skin_Cancer/ISIC-2017_Validation_Part1_GroundTruth/"
    as_gray = False
    output_shape = (256, 256)

    for dir in os.listdir(img_path):

        if 'crop_norm' in dir:
            corr_mask_dir = dir[:-14] + '_segmentation.png'
            l = os.listdir(mask_path)
            if corr_mask_dir in os.listdir(mask_path):
                print(dir)

                img = io.imread(os.path.join(img_path, dir),
                                as_gray=as_gray).astype(np.float32)
                img = normalize_pixel_values(img)
                img = transform.resize(img, output_shape, anti_aliasing=True, mode='constant')
                img = normalize_pixel_values(img)
                new_dir = dir[:-14] + '.jpg'
                io.imsave(os.path.join(save_img_path, new_dir), img)

    for dir in os.listdir(mask_path):
        
        print(dir)

        big_img_path = img_path + dir[:-16] + 'norm.jpg'
        template_dir = img_path + dir[:-16] + 'crop_norm.jpg'

        mask = crop_mask(img_dir=big_img_path, mask_dir=mask_path + dir, template_dir=template_dir)
        mask = transform.resize(mask, output_shape, anti_aliasing=True, mode='constant')
        new_dir = dir[:-17] + '.png'
        io.imsave(os.path.join(save_mask_path, new_dir), mask)


# resize_and_save('data/validation/image', 'data/validation/mask')
