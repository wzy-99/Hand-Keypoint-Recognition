import paddle
from paddleslim.dygraph import L1NormFilterPruner
from dataset import TrainDataset
from resnet import ResNet
from unet import Unet
import config


def loss(x, label):
    loss = x - label
    loss = 0.5 * loss * loss
    loss = loss * paddle.exp(config.WEIGHT * label)
    loss = paddle.mean(loss)
    return loss


def train():
    net = Unet(class_number=config.CLASS_NUMBER)
    model = paddle.Model(net)
    model.load('output/unet1')
    model.prepare(loss=loss)
    pruner = L1NormFilterPruner(net, [1, 3, 512, 512], "./usen.pickle")
    plan = pruner.sensitive_prune(0.6, skip_vars=['conv2d_18.w_0'])
    train_dataset = TrainDataset('./mydata')
    callback = paddle.callbacks.LRScheduler(by_step=True, by_epoch=False)
    scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=0.01, warmup_steps=len(train_dataset), start_lr=0, end_lr=0.01, verbose=True)
    # scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.2, T_max=1600*4, verbose=True)
    # optimizer = paddle.optimizer.SGD(learning_rate=scheduler, parameters=model.parameters())
    optimizer = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())
    model.prepare(optimizer, loss)
    model.fit(train_dataset, batch_size=4, epochs=20, callbacks=callback, drop_last=True)
    model.save('output/slimunet1')


if __name__ == "__main__":
    train()