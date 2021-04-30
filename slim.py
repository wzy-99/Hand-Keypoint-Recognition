import paddle
from paddle.vision.transforms import Compose, Resize, Transpose, Normalize
from paddleslim.dygraph import L1NormFilterPruner
from paddleslim.analysis import dygraph_flops
from resnet import ResNetv1
import config
from train import loss_v1
from dataset import TrainDataset_v1


transform = Compose([
    Resize(size=(config.INPUT_SIZE, config.INPUT_SIZE)),
    Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], data_format='HWC'),
    Transpose(),
])


def loss(x, label):
    loss = x - label
    loss = 0.5 * loss * loss
    loss = paddle.mean(loss)
    return loss


def slim():
    net = ResNetv1(class_dim=config.CLASS_NUMBER)

    model = paddle.Model(net)
    model.load('output/resnetv13')
    # model.prepare(metrics=paddle.metric.Accuracy())
    model.prepare(loss=loss)

    pruner = L1NormFilterPruner(net, [1, 3, config.INPUT_SIZE, config.INPUT_SIZE])

    valid_dataset = TrainDataset_v1('./myvalid')

    def eval_fn():
        result = model.evaluate(
            valid_dataset,
            batch_size=16)
        acc = 1.0 - result['loss'][0]
        return acc

    pruner.sensitive(eval_func=eval_fn, sen_file="./sen.pickle")

    print(pruner.sensitive())

    flops = dygraph_flops(net, [1, 3, config.INPUT_SIZE, config.INPUT_SIZE])

    plan = pruner.sensitive_prune(0.4)

    flops = dygraph_flops(net, [1, 3, config.INPUT_SIZE, config.INPUT_SIZE])
    print(f"FLOPs after pruning: {flops}")

    result = model.evaluate(val_dataset, batch_size=16, log_freq=10)
    print(f"before fine-tuning: {result}")

    model.save('output/slimmodel')


if __name__ == '__main__':
    slim()