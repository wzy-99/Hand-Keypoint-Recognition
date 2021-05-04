import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, BatchNorm, Pool2D, Dropout


class ConvBN(fluid.dygraph.Layer):
    def __init__(self, in_dim, out_dim, act='relu'):
        super(ConvBN, self).__init__()
        self.conv = Conv2D(num_channels=in_dim, num_filters=out_dim, filter_size=3, stride=1, padding=1)
        self.bn = BatchNorm(num_channels=out_dim, act=act)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ConvBlock(fluid.dygraph.Layer):
    def __init__(self, in_dim, out_dim, act='relu', down_sample=True, p=0.5):
        super(ConvBlock, self).__init__()

        if down_sample:
            self.down_sample = Pool2D(pool_size=2, pool_stride=2, pool_type="max")
        else:
            self.down_sample = None
        
        self.conv_bn1 = ConvBN(in_dim=in_dim, out_dim=out_dim, act=act)
        self.conv_bn2 = ConvBN(in_dim=out_dim, out_dim=out_dim, act=act)
        self.drop_out = Dropout(p=p)

    def forward(self, x):
        if self.down_sample:
            x = self.down_sample(x)
        x = self.conv_bn1(x)
        x = self.conv_bn2(x)
        x = self.drop_out(x)
        return x

# 用普通方法实现Unet
class SimpleUnet(fluid.dygraph.Layer):
    def __init__(self, class_number, in_dim=3, act='relu', p=0.5):
        super(SimpleUnet, self).__init__()
        
        self.block1 = ConvBlock(in_dim, 64 * 2 ** 0, act=act, down_sample=False, p=p)
        self.block2 = ConvBlock(64 * 2 ** 0, 64 * 2 ** 1, act=act, down_sample=True, p=p)
        self.block3 = ConvBlock(64 * 2 ** 1, 64 * 2 ** 2, act=act, down_sample=True, p=p)
        self.block4 = ConvBlock(64 * 2 ** 2, 64 * 2 ** 3, act=act, down_sample=True, p=p)
        self.block5 = ConvBlock(64 * 2 ** 3, 64 * 2 ** 4, act=act, down_sample=True, p=p)

        self.reverse_block1 = ConvBlock(64 * 2 ** 1 + 64 * 2 ** 0, 64 * 2 ** 0, act=act, down_sample=False, p=p)
        self.reverse_block2 = ConvBlock(64 * 2 ** 2 + 64 * 2 ** 1, 64 * 2 ** 1, act=act, down_sample=False, p=p)
        self.reverse_block3 = ConvBlock(64 * 2 ** 3 + 64 * 2 ** 2, 64 * 2 ** 2, act=act, down_sample=False, p=p)
        self.reverse_block4 = ConvBlock(64 * 2 ** 4 + 64 * 2 ** 3, 64 * 2 ** 3, act=act, down_sample=False, p=p)

        self.out = ConvBN(64 * 2 ** 0, class_number, act=act)

    def forward(self, x):
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        f5 = self.block5(f4)

        f4_reverse = fluid.layers.image_resize(f5, scale=2.0, resample='BILINEAR')
        f4_reverse = fluid.layers.concat([f4_reverse, f4], axis=1)
        f4_reverse = self.reverse_block4(f4_reverse)

        f3_reverse = fluid.layers.image_resize(f4_reverse, scale=2.0, resample='BILINEAR')
        f3_reverse = fluid.layers.concat([f3_reverse, f3], axis=1)
        f3_reverse = self.reverse_block3(f3_reverse)

        f2_reverse = fluid.layers.image_resize(f3_reverse, scale=2.0, resample='BILINEAR')
        f2_reverse = fluid.layers.concat([f2_reverse, f2], axis=1)
        f2_reverse = self.reverse_block2(f2_reverse)

        f1_reverse = fluid.layers.image_resize(f2_reverse, scale=2.0, resample='BILINEAR')
        f1_reverse = fluid.layers.concat([f1_reverse, f1], axis=1)
        f1_reverse = self.reverse_block1(f1_reverse)

        out = self.out(f1)

        return out

# 用更紧凑的写法实现Unet
class Unet(fluid.dygraph.Layer):
    def __init__(self, class_number, depth=5, in_dim=3, act='relu', p=0.5):
        super(Unet, self).__init__()

        self.depth = depth

        self.blocks = []
        for i in range(depth):
            if i == 0:
                self.blocks.append(ConvBlock(in_dim, 64 * 2 ** i, act=act, down_sample=False, p=p))
            else:
                self.blocks.append(ConvBlock(64 * 2 ** (i - 1), 64 * 2 ** i, act=act, down_sample=True, p=p))
        self.blocks = paddle.nn.LayerList(self.blocks)

        self.reverse_blocks = []
        for i in range(depth - 1, 0, -1):
            self.reverse_blocks.append(ConvBlock(64 * 2 ** i + 64 * 2 ** (i - 1), 64 * 2 ** (i - 1), act=act, down_sample=False, p=p))
        self.reverse_blocks = paddle.nn.LayerList(self.reverse_blocks)

        self.out = ConvBN(64 * 2 ** 0, class_number, act='sigmoid')

    def forward(self, x):
        f = [] 
        for i in range(self.depth):
            x = self.blocks[i](x)
            f.append(x)

        for i in range(self.depth - 1):
            x = fluid.layers.image_resize(x, scale=2.0, resample='BILINEAR')
            x = fluid.layers.concat([x, f[self.depth - i - 2]], axis=1)
            x = self.reverse_blocks[i](x)
        
        x = self.out(x)

        return x


if __name__ == "__main__":
    with fluid.dygraph.guard():
        # network = SimpleUnet(class_number=1) # depth = 5
        # paddle.summary(network, (1, 3, 512, 512))
        network = Unet(class_number=1, depth=5)
        paddle.summary(network, (1, 3, 512, 512))