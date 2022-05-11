import cv2
import numpy as np


def image_processing(img1, img2):
    #  resize图片大小，入口参数为一个tuple，新的图片的大小
    img_1 = np.resize(img1, (520, 520))
    img_2 = np.resize(img2, (520, 520))
    #  处理图片后存储路径，以及存储格式
    return img_1, img_2


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    img1, img2: [0, 255]
    '''
    img_1 = img1
    img_2 = img2
    if not img1.shape == img2.shape:
        img1, img2 = image_processing(img_1, img_2)
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
def change_SSIM(score):
    socre_change = 1-(1-score)/2
    return socre_change


content = cv2.imread("C:/Users/admin/Desktop/used_img/content/forest2.jpg", 0)
style = cv2.imread("C:/Users/admin/Desktop/used_img/style/candy.jpg", 0)
sys_img = cv2.imread("C:/Users/admin/Desktop/used_img/sys/SANet/compare.jpg", 0)
ss1 = calculate_ssim(sys_img, content)
ss2 = calculate_ssim(sys_img, style)
print(ss1)
print(ss2)
print("内容相似度为:", change_SSIM(ss1))
print("风格相似度为:", change_SSIM(ss2))




