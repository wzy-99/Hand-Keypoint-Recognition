import os
import cv2
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.inference as paddle_infer
from paddle.vision.transforms import Compose, Resize, Transpose, Normalize
import config

transform = Compose([
    Resize(size=(config.INPUT_SIZE, config.INPUT_SIZE)),
    Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], data_format='HWC'),
    Transpose(),
])

def create_predictor(model_dir,
                     use_gpu=True,
                     gpu_id=0,
                     use_mkl=False,
                     use_trt=False,
                     use_glog=False,
                     memory_optimize=True):
    cfg = fluid.core.AnalysisConfig(
            os.path.join(model_dir, '__model__'),
            os.path.join(model_dir, '__params__'))

    if use_gpu:
        # 设置GPU初始显存(单位M)和Device ID
        cfg.enable_use_gpu(100, gpu_id)
    else:
        cfg.disable_gpu()
    if use_mkl:
        cfg.enable_mkldnn()
    if use_glog:
        cfg.enable_glog_info()
    else:
        cfg.disable_glog_info()
    if memory_optimize:
        cfg.enable_memory_optim()

    # 开启计算图分析优化，包括OP融合等
    cfg.switch_ir_optim(True)
    # 关闭feed和fetch OP使用，使用ZeroCopy接口必须设置此项
    cfg.switch_use_feed_fetch_ops(False)
    predictor = fluid.core.create_paddle_predictor(cfg)
    return predictor

def infer():
    # 读取模型文件
    # 格式为是 __model__ __params__
    predictor = create_predictor('model2')

    img = cv2.imread('test/1.jpg')
    img = transform(img).astype('float32')
    img = img[np.newaxis, :]

    # 获取输入的名称
    input_names = predictor.get_input_names()
    # 获取输入的tensor
    input_tensor = predictor.get_input_tensor(input_names[0])
    # 将图像放入
    input_tensor.copy_from_cpu(img)
    # 开始运行
    predictor.zero_copy_run()
    # 得到结果
    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_tensor(output_names[0])
    result = output_tensor.copy_to_cpu().reshape(config.LABEL_SIZE, config.LABEL_SIZE)

    cv2.imshow('result', result)
    cv2.waitKey(0)

if __name__ == '__main__':
    infer()