import numpy as np
import paddle
import cv2
import config
from dataset import TestDataset
from resnet import ResNetv1, ResNetv2


def predict_v1():
    testdataset = TestDataset('test')
    net = ResNetv1(class_dim=config.CLASS_NUMBER)
    model = paddle.Model(net)
    model.load('output/resnet1')
    model.prepare()
    outs = model.predict(testdataset)
    outs = outs[0]
    for index, out in enumerate(outs):
        image_path = testdataset.indexs[index]
        print(image_path)
        res = np.reshape(out, (224, 224))
        y, x = np.unravel_index(res.argmax(), res.shape)
        print('xy', x, y)
        min_value = res.min()
        max_value = res.max()
        mean_value = res.mean()
        print('minmaxmean', min_value, max_value, mean_value)
        res = res - min_value
        img = cv2.imread(image_path)
        h, w, c = img.shape
        img = cv2.circle(img, (int(x * w / 224), int(y * h / 224)), 5, (0, 0, 255), -1)
        res = res / max_value * 254
        res = res.astype(np.uint8)
        cv2.imwrite('result/' + str(index) + 'result.jpg', res)
        cv2.imwrite('result/' + str(index) + 'origin.jpg', img)


if __name__ == '__main__':
    predict_v1()