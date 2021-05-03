import paddle
import paddle.fluid as fluid
from paddle.vision.transforms import Compose, Resize, Transpose, Normalize
from paddleslim.dygraph import L1NormFilterPruner
from paddleslim.analysis import dygraph_flops
from resnet import ResNet
from dataset import TrainDataset
import config


transform = Compose([
    Resize(size=(config.INPUT_SIZE, config.INPUT_SIZE)),
    Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], data_format='HWC'),
    Transpose(),
])


def loss(x, label):
    loss = x - label
    loss = 0.5 * loss * loss
    loss = loss * (label > 0).astype('float32')
    loss = paddle.mean(loss)
    return loss


def slim():
    net = ResNet(class_dim=config.CLASS_NUMBER)

    # state, _ = fluid.load_dygraph('output/resnetv13')
    state, _ = fluid.load_dygraph('output/slimmodel')

    net.set_state_dict(state)

    for param in net.parameters():
        print(param.shape)

    # model = paddle.Model(net)
    # model.load('output/resnetv13')
    # model.prepare(loss=loss)
    # model.prepare()

    paddle.summary(net, (1, 3, config.INPUT_SIZE, config.INPUT_SIZE))

    pruner = L1NormFilterPruner(net, [1, 3, config.INPUT_SIZE, config.INPUT_SIZE], "./sen.pickle")
    # pruner = L1NormFilterPruner(net, [1, 3, config.INPUT_SIZE, config.INPUT_SIZE])

    # valid_dataset = TrainDataset('./myvalid')

    # def eval_fn():
    #     result = model.evaluate(
    #         valid_dataset,
    #         batch_size=1)
    #     acc = 1.0 - result['loss'][0]
    #     return acc

    # pruner.sensitive(eval_func=eval_fn, sen_file="./sen.pickle")

    # print(pruner.sensitive())

    flops = dygraph_flops(net, [1, 3, config.INPUT_SIZE, config.INPUT_SIZE])
    print(f"FLOPs before pruning: {flops}")

    plan = pruner.sensitive_prune(0.4)
    print(f"plan: {plan}")

    flops = dygraph_flops(net, [1, 3, config.INPUT_SIZE, config.INPUT_SIZE])
    print(f"FLOPs after pruning: {flops}")

    paddle.summary(net, (1, 3, config.INPUT_SIZE, config.INPUT_SIZE))

    # result = model.evaluate(valid_dataset, batch_size=1, log_freq=10)
    # print(f"after fine-tuning: {result}")

    # model.save('output/slimmodel')

    for param in net.parameters():
        print(param.shape)

    state_dict = net.state_dict()

    # paddle.save(state_dict, "paddle_dy.pdparams")

if __name__ == '__main__':
    slim()