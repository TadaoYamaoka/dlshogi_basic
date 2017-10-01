import numpy as np
from chainer import Chain
import chainer.functions as F
import chainer.links as L

from dlshogi_basic.common import *

k = 192
class PolicyNetwork(Chain):
    def __init__(self):
        super(PolicyNetwork, self).__init__(
            l1=L.Convolution2D(in_channels = None, out_channels = k, ksize = 3, pad = 1),
            l2=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1),
            l3=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1),
            l4=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1),
            l5=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1),
            l6=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1),
            l7=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1),
            l8=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1),
            l9=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1),
            l10=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1),
            l11=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1),
            l12=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1),
            l13=L.Convolution2D(in_channels = k, out_channels = MOVE_DIRECTION_LABEL_NUM, ksize = 1, nobias = True),
            l13_2=L.Bias(shape=(9*9*MOVE_DIRECTION_LABEL_NUM))
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        h5 = F.relu(self.l5(h4))
        h6 = F.relu(self.l6(h5))
        h7 = F.relu(self.l7(h6))
        h8 = F.relu(self.l8(h7))
        h9 = F.relu(self.l9(h8))
        h10 = F.relu(self.l10(h9))
        h11 = F.relu(self.l11(h10))
        h12 = F.relu(self.l12(h11))
        h13 = self.l13(h12)
        return self.l13_2(F.reshape(h13, (len(h13.data), 9*9*MOVE_DIRECTION_LABEL_NUM)))
