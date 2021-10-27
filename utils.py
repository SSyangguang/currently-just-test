import torch
import torch.nn.functional as F


def pad_to(x, stride):
    '''
    For encoder of U-Net model, padding to avoid odd resolutions
    https://stackoverflow.com/questions/66028743/how-to-handle-odd-resolutions-in-unet-architecture-pytorch
    '''
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pads = (lw, uw, lh, uh)

    out = F.pad(x, pads, "constant", 0)

    return out, pads


def unpad(x, pad):
    '''
    For decoder of U-Net model, unpadding
    https://stackoverflow.com/questions/66028743/how-to-handle-odd-resolutions-in-unet-architecture-pytorch
    '''
    if pad[2]+pad[3] > 0:
        x = x[:, :, pad[2]:-pad[3], :]
    if pad[0]+pad[1] > 0:
        x = x[:, :, :, pad[0]:-pad[1]]
    return x
