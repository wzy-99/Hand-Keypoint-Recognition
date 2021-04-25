import numpy as np
from scipy import signal


hotmap_template = {}


def hotmap(img, P, R, sigma=(1.0, 1.0)):
    h, w = img.shape
    cx, cy = P
    cx, cy = int(cx), int(cy)
    sigma_x, sigma_y = sigma
    if (h, w) in hotmap_template:
        guass_window = hotmap_template[(h ,w)]
    else:
        guass_h = signal.windows.gaussian(2*h, R*sigma_y)
        guass_W = signal.windows.gaussian(2*w, R*sigma_x)
        guass_window = np.multiply(guass_h.reshape(-1, 1), guass_W)
        hotmap_template[(h, w)] = guass_window
    return np.clip(img + guass_window[h-cy:2*h-cy, w-cx:2*w-cx], 0.0, 1.0)

