import paddle
from dataset import TrainDataset_v1 
from resnet import ResNetv1
import config


def loss_v1(x, label):
    # loss = paddle.nn.functional.square_error_cost(x, label)
    # return loss
    loss = x - label
    loss = 0.5 * loss * loss
    loss = loss * paddle.exp(config.WEIGHT * label)
    loss = paddle.mean(loss)
    return loss


# def loss_v2(x, label):
#     loss = paddle.nn.functional.square_error_cost(x, label)
#     loss = paddle.mean(loss)
#     return loss


def train_v1():
    net = ResNetv1(class_dim=config.CLASS_NUMBER)
    model = paddle.Model(net)
    # model.load('output/resnet1')
    train_dataset = TrainDataset_v1('./mydata')
    callback = paddle.callbacks.LRScheduler(by_step=True, by_epoch=False)
    scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=0.01, warmup_steps=len(train_dataset), start_lr=0, end_lr=0.01, verbose=True)
    # scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.2, T_max=1600*4, verbose=True)
    # optimizer = paddle.optimizer.SGD(learning_rate=scheduler, parameters=model.parameters())
    optimizer = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())
    model.prepare(optimizer, loss_v1)
    model.fit(train_dataset, batch_size=1, epochs=4, callbacks=callback, drop_last=True)
    model.save('output/resnet1')


# def train_v2():
#     net = ResNetv2(class_dim=config.CLASS_NUMBER)
#     model = paddle.Model(net)
#     # model.load('output/resnet1')
#     train_dataset = TrainDataset_v2('./mydata')
#     callback = paddle.callbacks.LRScheduler(by_step=True, by_epoch=False)
#     scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=0.001, warmup_steps=1000, start_lr=0, end_lr=0.001, verbose=True)
#     # scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.2, T_max=1600*4, verbose=True)
#     # optimizer = paddle.optimizer.SGD(learning_rate=scheduler, parameters=model.parameters())
#     optimizer = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())
#     model.prepare(optimizer, loss_v2)
#     model.fit(train_dataset, batch_size=1, epochs=4, callbacks=callback, drop_last=True)
#     # model.save('output/resnet1')



if __name__ == "__main__":
    train_v1()