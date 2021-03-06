import numpy as np
import paddle
import cv2
import config
from dataset import TestDataset
from resnet import ResNet


def predict():
    testdataset = TestDataset('test')
    net = ResNet(class_dim=config.CLASS_NUMBER)
    model = paddle.Model(net)
    model.load('output/resnetv11')
    model.prepare()
    outs = model.predict(testdataset)
    outs = outs[0]
    for index, out in enumerate(outs):
        image_path = testdataset.indexs[index]
        print(image_path)
        res = np.reshape(out, (config.LABEL_SIZE, config.LABEL_SIZE))
        y, x = np.unravel_index(res.argmax(), res.shape)
        print('xy', x, y)
        min_value = res.min()
        max_value = res.max()
        mean_value = res.mean()
        print('min max mean', min_value, max_value, mean_value)
        img = cv2.imread(image_path)
        h, w, c = img.shape
        img = cv2.circle(img, (int(x * w / config.LABEL_SIZE), int(y * h / config.LABEL_SIZE)), 5, (0, 0, 255), -1)
        res = res * 255
        res = res.astype(np.uint8)
        cv2.imwrite('result/' + str(index) + 'result.jpg', res)
        cv2.imwrite('result/' + str(index) + 'origin.jpg', img)


if __name__ == '__main__':
    predict()