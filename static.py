import numpy as np
import paddle
import paddle.fluid as fluid
from paddleslim.dygraph import L1NormFilterPruner
from paddleslim.analysis import dygraph_flops
from paddle.jit import TracedLayer
from resnet import ResNet
import config


def static():
    net = ResNet(class_dim=config.CLASS_NUMBER)

    net.eval()

    state, _ = fluid.load_dygraph('output/slimmodel')

    net.set_state_dict(state)

    in_np = np.random.random([1, 3, config.INPUT_SIZE, config.INPUT_SIZE]).astype('float32')

    input_var = paddle.to_tensor(in_np)

    # pruner = L1NormFilterPruner(net, [1, 3, config.INPUT_SIZE, config.INPUT_SIZE], "./sen.pickle")

    # plan = pruner.sensitive_prune(0.4)

    paddle.jit.save(net, "inference_model", input_spec=[input_var])


if __name__ == '__main__':
    static()