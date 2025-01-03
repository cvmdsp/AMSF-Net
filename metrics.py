import torch
import numpy as np
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
def calc_ergas(img_tgt, img_fus):
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    errors = np.abs(img_tgt - img_fus)
    # 计算绝对误差的平均值
    moae = np.mean(errors)
    # img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    # img_fus = img_fus.reshape(img_fus.shape[0], -1)

    # rmse = np.mean((img_tgt-img_fus)**2, axis=1)
    rmse = moae**0.5
    mean = np.mean(img_fus)

    ergas = np.mean((rmse/mean)**2)
    ergas = 100/110*ergas**0.5

    return ergas

def calc_psnr(img_tgt, img_fus):

    mse = np.mean((img_tgt-img_fus)**2)
    img_max = np.max(img_tgt)
    psnr = 10*np.log10(img_max**2/mse)

    return psnr

def calc_rmse(img_tgt, img_fus):
    rmse = np.sqrt(np.mean((img_tgt-img_fus)**2))

    return rmse

def calc_sam(x_true, x_pred):
    x_true = x_true / 255.0
    x_pred = x_pred / 255.0
    x_true = np.squeeze(x_true)
    x_pred = np.squeeze(x_pred)
    img1 = np.transpose(x_true, (1, 2, 0))
    img2 = np.transpose(x_pred, (1, 2, 0))
    # 输入校验
    assert img1.ndim == 3 and img2.ndim == 3
    assert img1.shape == img2.shape

    H, W, C = img1.shape

    # 归一化到 [0,1]
    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())

    # 计算向量点积和范数
    dot_product = np.sum(img1 * img2, axis=-1)
    norm_img1 = np.linalg.norm(img1, axis=-1)
    norm_img2 = np.linalg.norm(img2, axis=-1)

    # 计算余弦相似度
    similarity = dot_product / (norm_img1 * norm_img2)

    # 对无效值进行处理
    similarity[np.isnan(similarity)] = 0

    # 求平均值得到SAM分数
    sam = np.arccos(similarity).mean()*180

    return sam


def calc_cc(img_tgt, img_fus):
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img1 = np.transpose(img_tgt, (1, 2, 0))
    img2 = np.transpose(img_fus, (1, 2, 0))

    assert img1.shape == img2.shape

    # 将图像展平为一维
    img1 = img1.reshape(-1)
    img2 = img2.reshape(-1)

    # 计算CC
    cc = np.corrcoef(img1, img2)[0, 1]
    return np.mean(cc)

def calc_moae(img_tgt, img_fus):
    # 计算图像每个像素的绝对误差
    errors = np.abs(img_tgt - img_fus)
    # 计算绝对误差的平均值
    moae = np.mean(errors)

    return moae

def calc_uiqi(img_tgt, img_fus):
     img_tgt = np.squeeze(img_tgt)
     img_tgt = np.transpose(img_tgt, (1, 2, 0))
     img_fus = np.squeeze(img_fus)
     img_fus = np.transpose(img_fus, (1, 2, 0))
     u_x = np.mean(img_tgt, axis=(0,1))
     u_y = np.mean(img_fus, axis=(0,1))

     sig_x = np.std(img_tgt, axis=(0,1))
     sig_y = np.std(img_fus, axis=(0,1))

     channels = img_tgt.shape[2]
     sig_xy = np.zeros(channels)
     for ch in range(channels):
         sig_xy[ch] = np.sum((img_tgt[:, :, ch] - u_x[ch]) * (img_fus[:, :, ch] - u_y[ch])) / (len(img_tgt) ** 2 - 1)
     # if sig_x == 0 or sig_y == 0:
     #     sig_x = sig_y = 1e-8
     uiqi = np.zeros(channels)
     for ch in range(channels):
         num = 4 * sig_xy[ch] * u_x[ch] * u_y[ch]
         den = (sig_x[ch] ** 2 + sig_y[ch] ** 2) * (u_x[ch] ** 2 + u_y[ch] ** 2)
         uiqi[ch] = num / den
     uiqi = np.mean(np.mean(uiqi, axis=0))
     return uiqi


def calc_ssim(img_tgt, img_fus):
    img_tgt = np.squeeze(img_tgt)
    img1 = np.transpose(img_tgt, (1, 2, 0))
    img_fus = np.squeeze(img_fus)
    img2 = np.transpose(img_fus, (1, 2, 0))
    ssim_index = structural_similarity(img1, img2, channel_axis=-1)
    # ssim返回4个值
    ssim = ssim_index


    return ssim