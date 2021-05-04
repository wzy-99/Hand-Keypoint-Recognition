import paddle
import paddle.fluid as fluid
from paddle.vision.transforms import Compose, Resize, Transpose, Normalize
from paddleslim.dygraph import L1NormFilterPruner
from paddleslim.analysis import dygraph_flops
from resnet import ResNet
from unet import Unet
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
    # loss = loss * (label > 0.5).astype('float32')
    loss = loss * paddle.exp(config.WEIGHT * label)
    loss = paddle.mean(loss)
    return loss


def slim():
    # net = ResNet(class_dim=config.CLASS_NUMBER)
    net = Unet(class_number=config.CLASS_NUMBER)

    model = paddle.Model(net)

    # model.load('output/resnet2')
    model.load('output/unet1')
    model.prepare(loss=loss)

    for param in net.parameters():
        print(param.name, param.shape)

    paddle.summary(net, (1, 3, config.INPUT_SIZE, config.INPUT_SIZE))

    flop, flops = dygraph_flops(net, [1, 3, config.INPUT_SIZE, config.INPUT_SIZE], detail=True)
    print('flop', flop)
    for k, v in flops.items():
        print(k, v)

    print('\n' * 5)

    pruner = L1NormFilterPruner(net, [1, 3, config.INPUT_SIZE, config.INPUT_SIZE], "./usen.pickle")
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

    plan = pruner.sensitive_prune(0.6, skip_vars=['conv2d_18.w_0'])
    # print(f"plan: {plan}")

    for param in net.parameters():
        print(param.name, param.shape)

    flop, flops = dygraph_flops(net, [1, 3, config.INPUT_SIZE, config.INPUT_SIZE], detail=True)
    print('flop', flop)
    for k, v in flops.items():
        print(k, v)
    
    paddle.summary(net, (1, 3, config.INPUT_SIZE, config.INPUT_SIZE))
    
    # result = model.evaluate(valid_dataset, batch_size=1, log_freq=10)
    # print(f"after fine-tuning: {result}")

    model.save('output/inference_model')

if __name__ == '__main__':
    slim()