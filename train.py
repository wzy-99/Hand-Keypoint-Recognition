import paddle.fluid as fluid
import numpy as np
import paddle
from dataset import TrainDatasetA, TrainDatasetB
from resnet import ResNet
import config


def loss_v1(x, label):
    loss = paddle.nn.functional.square_error_cost(x, label)
    return loss


def loss_v2(x, label):
    loss = paddle.nn.functional.square_error_cost(x, label)
    return loss


def train_v1():
    net = ResNet()
    model = paddle.Model(net)
    # model.load('output/resnet1')
    train_dataset = TrainDatasetA('./mydata', class_number=config.CLASS_NUMBER)
    callback = paddle.callbacks.LRScheduler(by_step=True, by_epoch=False)
    scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=0.01, warmup_steps=400, start_lr=0, end_lr=0.01, verbose=True)
    # scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.2, T_max=1600*4, verbose=True)
    optimizer = paddle.optimizer.SGD(learning_rate=scheduler, parameters=model.parameters())
    model.prepare(optimizer, loss_v1)
    model.fit(train_dataset, batch_size=32, epochs=4, callbacks=callback)
    # model.save('output/resnet1')


def train_v2():
    net = ResNet()
    model = paddle.Model(net)
    # model.load('output/resnet1')
    train_dataset = TrainDatasetA('./mydata', class_number=config.CLASS_NUMBER)
    callback = paddle.callbacks.LRScheduler(by_step=True, by_epoch=False)
    scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=0.01, warmup_steps=400, start_lr=0, end_lr=0.01, verbose=True)
    # scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.2, T_max=1600*4, verbose=True)
    optimizer = paddle.optimizer.SGD(learning_rate=scheduler, parameters=model.parameters())
    model.prepare(optimizer, loss_v2)
    model.fit(train_dataset, batch_size=32, epochs=4, callbacks=callback)
    # model.save('output/resnet1')



if __name__ == "__main__":
    train()