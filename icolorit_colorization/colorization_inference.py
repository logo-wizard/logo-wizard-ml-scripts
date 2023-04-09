import datetime
import glob
import os
import sys
import cv2
import numpy as np
import torch
from einops import rearrange
from skimage import color
import warnings
from timm.models import create_model
import torch
import modeling

class Gamut():
    def __init__(self, gamut_size=110):
        self.gamut_size = gamut_size
        self.win_size = gamut_size * 2
        self.ab_grid = Grid(gamut_size=gamut_size, D=1)

    def set_gamut(self, l_in=50):
        self.l_in = l_in
        self.ab_map, self.mask = self.ab_grid.update_gamut(l_in=l_in)
        ab_map = cv2.resize(self.ab_map, (self.win_size, self.win_size))

    def set_ab(self, color):
        self.color = color
        self.lab = rgb2lab_1d(self.color)
        x, y = self.ab_grid.ab2xy(self.lab[1], self.lab[2])
        self.pos = (x, y)

    def is_valid_point(self, pos):
        if pos is None or self.mask is None:
            return False
        else:
            x = pos.x()
            y = pos.y()
            if x >= 0 and y >= 0 and x < self.win_size and y < self.win_size:
                return self.mask[y, x]
            else:
                return False

    def update(self, pos):
        self.pos = pos
        a, b = self.ab_grid.xy2ab(pos[0], pos[1])
        L = self.l_in
        lab = np.array([L, a, b])
        color = lab2rgb_1d(lab, clip=True, dtype='uint8')
        return color


class Grid():
    def __init__(self, gamut_size=110, D=1):
        self.D = D
        self.vals_b, self.vals_a = np.meshgrid(np.arange(-gamut_size, gamut_size + D, D),
                                               np.arange(-gamut_size, gamut_size + D, D))
        self.pts_full_grid = np.concatenate((self.vals_a[:, :, np.newaxis], self.vals_b[:, :, np.newaxis]), axis=2)
        self.A = self.pts_full_grid.shape[0]
        self.B = self.pts_full_grid.shape[1]
        self.gamut_size = gamut_size

    def update_gamut(self, l_in):
        warnings.filterwarnings("ignore")
        thresh = 1.0
        pts_lab = np.concatenate((l_in + np.zeros((self.A, self.B, 1)), self.pts_full_grid), axis=2)
        self.pts_rgb = (255 * np.clip(color.lab2rgb(pts_lab), 0, 1)).astype('uint8')
        pts_lab_back = color.rgb2lab(self.pts_rgb)
        pts_lab_diff = np.linalg.norm(pts_lab - pts_lab_back, axis=2)

        self.mask = pts_lab_diff < thresh
        mask3 = np.tile(self.mask[..., np.newaxis], [1, 1, 3])
        self.masked_rgb = self.pts_rgb.copy()
        self.masked_rgb[np.invert(mask3)] = 255
        return self.masked_rgb, self.mask

    def ab2xy(self, a, b):
        y = self.gamut_size + a
        x = self.gamut_size + b
        return x, y

    def xy2ab(self, x, y):
        a = y - self.gamut_size
        b = x - self.gamut_size
        return a, b


class Colorization():
    def __init__(self, model, load_size=224, win_size=512, device='cpu'):
        self.image_file = None
        self.pos = None
        self.model = model
        self.win_size = win_size
        self.load_size = load_size
        self.device = device
        self.im_gray3 = None
        self.poses = []
        self.colors = []
        self.gamut = Gamut(gamut_size=110)

    def read_image(self, image_file):
        self.image_file = image_file
        im_bgr = cv2.imread(image_file)
        self.im_full = im_bgr.copy()

        h, w, c = self.im_full.shape
        max_width = max(h, w)
        r = self.win_size / float(max_width)
        self.scale = float(self.win_size) / self.load_size
        rw = int(round(r * w / 4.0) * 4)
        rh = int(round(r * h / 4.0) * 4)

        self.im_win = cv2.resize(self.im_full, (rw, rh), interpolation=cv2.INTER_CUBIC)

        self.dw = int((self.win_size - rw) // 2)
        self.dh = int((self.win_size - rh) // 2)
        self.win_w = rw
        self.win_h = rh
        im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
        self.im_gray3 = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)

        self.gray_win = cv2.resize(self.im_gray3, (rw, rh), interpolation=cv2.INTER_CUBIC)
        im_bgr = cv2.resize(im_bgr, (self.load_size, self.load_size), interpolation=cv2.INTER_CUBIC)
        self.im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        lab_win = color.rgb2lab(self.im_win[:, :, ::-1])

        self.im_lab = color.rgb2lab(im_bgr[:, :, ::-1])
        self.im_l = self.im_lab[:, :, 0]
        self.l_win = lab_win[:, :, 0]
        self.im_ab = self.im_lab[:, :, 1:]
        self.im_size = self.im_rgb.shape[0:2]

        self.im_ab0 = np.zeros((2, self.load_size, self.load_size))
        self.im_mask0 = np.zeros((1, self.load_size, self.load_size))
        self.brushWidth = 2 * self.scale

        self.scale_gamut = float(np.max((rw, rh))) / self.load_size


    def scale_point(self, pnt):
        x = int((pnt[0] - self.dw) / float(self.win_w) * self.load_size)
        y = int((pnt[1] - self.dh) / float(self.win_h) * self.load_size)
        return x, y

    def change_color(self, pos_gamut):
        x, y = self.scale_point(self.pos)
        L = self.im_lab[y, x, 0]
        self.gamut.set_gamut(l_in=L)
        color = self.gamut.update(pos_gamut)
        self.set_color(color)

    def calibrate_color(self, c, pos):
        x, y = self.scale_point(pos)

        color_array = np.array((c[0], c[1], c[2])).astype('uint8')
        mean_L = self.im_l[y, x]
        snap_color = snap_ab(mean_L, color_array) 
        return (snap_color[0], snap_color[1], snap_color[2])

    def scale_point_control(self, in_x, in_y, w):
        x = int((in_x - self.dw) / float(self.win_w) * self.load_size) + w
        y = int((in_y - self.dh) / float(self.win_h) * self.load_size) + w
        return x, y

    def update_image_mask(self, im, mask, pnt, clr):
        w = int(self.brushWidth / self.scale_gamut)
        x1, y1 = self.scale_point_control(pnt[0], pnt[1], -w)
        tl = (x1, y1)
        br = (x1 + 1, y1 + 1) # hint size fixed to 2
        c = (int(clr[0]), int(clr[1]), int(clr[2]))
        cv2.rectangle(mask, tl, br, 255, -1)
        cv2.rectangle(im, tl, br, c, -1)

        return im, mask

    def set_color(self, c_rgb):
        c = (c_rgb[0], c_rgb[1], c_rgb[2])
        self.user_color = c
        snap_qcolor = self.calibrate_color(c, self.pos)
        self.color = snap_qcolor
        self.colors.append(snap_qcolor)
        self.compute_result()

    def set_pos(self, pos):
        self.pos = pos
        self.poses.append(pos)

    def get_model_input(self):
        h = self.load_size
        w = self.load_size
        im = np.zeros((h, w, 3), np.uint8)
        mask = np.zeros((h, w, 1), np.uint8)

        for pnt, clr in zip(self.poses, self.colors):
            im, mask = self.update_image_mask(im, mask, pnt, clr)

        return im, mask

    def compute_result(self):
        im, mask = self.get_model_input()
        im_mask0 = mask > 0.0
        self.im_mask0 = im_mask0.transpose((2, 0, 1))  # (1, H, W)
        im_lab = color.rgb2lab(im).transpose((2, 0, 1))  # (3, H, W)
        self.im_ab0 = im_lab[1:3, :, :]

        # _im_lab is 1) normalized 2) a torch tensor
        _im_lab = self.im_lab.transpose((2,0,1))
        _im_lab = np.concatenate(((_im_lab[[0], :, :] - 50) / 100, _im_lab[1:, :, :] / 110), axis=0)
        _im_lab = torch.from_numpy(_im_lab).type(torch.FloatTensor).to(self.device)

        # _img_mask is 1) normalized ab 2) flipped mask
        _img_mask = np.concatenate((self.im_ab0 / 110, (255 - self.im_mask0) / 255), axis=0)
        _img_mask = torch.from_numpy(_img_mask).type(torch.FloatTensor).to(self.device)

        # _im_lab is the full color image, _img_mask is the ab_hint + mask
        ab = self.model(_im_lab.unsqueeze(0), _img_mask.unsqueeze(0))
        ab = rearrange(ab, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c', 
                        h=self.load_size//self.model.patch_size, w=self.load_size//self.model.patch_size,
                        p1=self.model.patch_size, p2=self.model.patch_size)[0]
        ab = ab.detach().numpy()

        ab_win = cv2.resize(ab, (self.win_w, self.win_h), interpolation=cv2.INTER_CUBIC)
        ab_win = ab_win * 110
        pred_lab = np.concatenate((self.l_win[..., np.newaxis], ab_win), axis=2)
        pred_rgb = (np.clip(color.lab2rgb(pred_lab), 0, 1) * 255).astype('uint8')
        self.result = pred_rgb


def get_model(model_path='icolorit_small_4ch_patch16_224.pth', device='cpu'):
    model = create_model(
        model_path[:-4],
        pretrained=False,
        drop_path_rate=0.0,
        drop_block_rate=None,
        use_rpb=True,
        avg_hint=True,
        head_mode='cnn',
        mask_cent=False,
    )
    model.to('cpu')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model


def rgb2lab_1d(in_rgb):
    return color.rgb2lab(in_rgb[np.newaxis, np.newaxis, :]).flatten()


def lab2rgb_1d(in_lab, clip=True, dtype='uint8'):
    tmp_rgb = color.lab2rgb(in_lab[np.newaxis, np.newaxis, :]).flatten()
    if clip:
        tmp_rgb = np.clip(tmp_rgb, 0, 1)
    if dtype == 'uint8':
        tmp_rgb = np.round(tmp_rgb * 255).astype('uint8')
    return tmp_rgb


def snap_ab(input_l, input_rgb, return_type='rgb'):
    ''' given an input lightness and rgb, snap the color into a region where l,a,b is in-gamut
    '''
    T = 20
    warnings.filterwarnings("ignore")
    input_lab = rgb2lab_1d(np.array(input_rgb))  # convert input to lab
    conv_lab = input_lab.copy()  # keep ab from input
    for t in range(T):
        conv_lab[0] = input_l  # overwrite input l with input ab
        old_lab = conv_lab
        tmp_rgb = color.lab2rgb(conv_lab[np.newaxis, np.newaxis, :]).flatten()
        tmp_rgb = np.clip(tmp_rgb, 0, 1)
        conv_lab = color.rgb2lab(tmp_rgb[np.newaxis, np.newaxis, :]).flatten()
        dif_lab = np.sum(np.abs(conv_lab - old_lab))
        if dif_lab < 1:
            break

    conv_rgb_ingamut = lab2rgb_1d(conv_lab, clip=True, dtype='uint8')
    if (return_type == 'rgb'):
        return conv_rgb_ingamut

    elif(return_type == 'lab'):
        conv_lab_ingamut = rgb2lab_1d(conv_rgb_ingamut)
        return conv_lab_ingamut


def main(img_path, model_path, win_size, device, poses, poses_gamut):
    color_model = get_model(model_path=model_path, device=device)
    colorization = Colorization(model=color_model, load_size=224, win_size=win_size, device=device)
    colorization.read_image(img_path)
    for pos, pos_gamut in zip(poses, poses_gamut):
        colorization.set_pos(pos)
        colorization.change_color(pos_gamut)
    cv2.imshow('result', cv2.cvtColor(colorization.result, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img_path = 'auxiliary/demo_image.jpg'
    model_path = 'auxiliary/icolorit_small_4ch_patch16_224.pth'
    win_size = 720
    device = 'cpu'
    pos = [
        (297, 269),
        (116, 258),
        (311, 75)
    ]
    pos_gamut = [
        (173, 175),
        (180, 88),
        (136, 75)
    ]
    main(img_path, model_path, win_size, device, pos, pos_gamut)