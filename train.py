import paddle
from dataset import TrainDataset
from resnet import ResNet
import config


def loss(x, label):
    loss = x - label
    loss = 0.5 * loss * loss
    loss = loss * paddle.exp(config.WEIGHT * label)
    loss = paddle.mean(loss)
    return loss



def train():
    net = ResNet(class_dim=config.CLASS_NUMBER)
    model = paddle.Model(net)
    # model.load('output/resnetv13')
    train_dataset = TrainDataset('./mydata')
    callback = paddle.callbacks.LRScheduler(by_step=True, by_epoch=False)
    scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=0.01, warmup_steps=len(train_dataset), start_lr=0, end_lr=0.01, verbose=True)
    # scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.2, T_max=1600*4, verbose=True)
    # optimizer = paddle.optimizer.SGD(learning_rate=scheduler, parameters=model.parameters())
    optimizer = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())
    model.prepare(optimizer, loss)
    model.fit(train_dataset, batch_size=1, epochs=4, callbacks=callback, drop_last=True)
    model.save('output/resnet1')


if __name__ == "__main__":
    train()