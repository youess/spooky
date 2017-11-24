# -*- coding: utf-8 -*-

"""
Text -> embedding -> NN classifier

                            conv(1) -> BN -> RELU -> conv(1) -> BN -> RELU -> MaxPooling
                            conv(2) -> BN -> RELU -> conv(2) -> BN -> RELU -> MaxPooling
Text Vector -> Embedding -> conv(3) -> BN -> RELU -> conv(3) -> BN -> RELU -> MaxPooling  -> concat -> softmax
                            conv(4) -> BN -> RELU -> conv(4) -> BN -> RELU -> MaxPooling
                            conv(5) -> BN -> RELU -> conv(5) -> BN -> RELU -> MaxPooling
"""


class TextCNN(object):
    pass
