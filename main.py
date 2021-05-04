import paddle.fluid as fluid
import numpy as np
import time
import paddle
import cv2
import paddleslim
from resnet import ResNet
from unet import Unet
from paddle.vision.transforms import Compose, Resize, Transpose, Normalize
from paddleslim.dygraph import L1NormFilterPruner
from paddleslim.analysis import dygraph_flops
import config


class Timer:
    def __init__(self):
        self.t = 0

    def __enter__(self):
        self.t = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(time.time() - self.t)


INPUT_SIZE = 512
LABEL_SIZE = 512

transform = Compose([
    Resize(size=(INPUT_SIZE, INPUT_SIZE)),
    Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], data_format='HWC'), 
    Transpose(), 
])


def init():
    # net = ResNet(class_dim=1)
    net = Unet(class_number=1)
    net.eval()

    # state, _ = fluid.load_dygraph('output/resnet2')
    state, _ = fluid.load_dygraph('output/slimunet1')

    net.set_state_dict(state)

    # for param in net.parameters():
    #     print(param.name, param.shape, '\n', param.numpy())

    # pruner = L1NormFilterPruner(net, [1, 3, 512, 512], "./sen.pickle")
    pruner = L1NormFilterPruner(net, [1, 3, 512, 512], "./usen.pickle")

    plan = pruner.sensitive_prune(0.6, skip_vars=['conv2d_18.w_0'])

    net.set_state_dict(state)

    for param in net.parameters():
        print(param.shape)

    # flops = dygraph_flops(net, [1, 3, config.INPUT_SIZE, config.INPUT_SIZE], detail=True)

    # print(flops)

    # paddle.summary(net, (1, 3, 512, 512))

    capture = cv2.VideoCapture(0)
    while True:
        ref, frame = capture.read()

        roi = frame.copy()
        h, w, _ = roi.shape
        roi = transform(roi).astype("float32")
        roi = roi[np.newaxis, :]
        roi = paddle.to_tensor(roi)

        with Timer():
            out = net(roi)
            out = out.numpy()

        res = np.reshape(out, (LABEL_SIZE, LABEL_SIZE))
        rh, rw = res.shape
        region = res[:, :]
        idx = np.argmax(region)
        y, x = np.unravel_index(region.argmax(), region.shape)
        cx = int(x / LABEL_SIZE * w)
        cy = int(y / LABEL_SIZE * h)

        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        cv2.imshow('gray', (res * 255).astype('uint8'))
        cv2.imshow("result", frame)
        c = cv2.waitKey(30) & 0xff
        if c == 27:
            capture.release()
            break


if __name__ == "__main__":
    init()