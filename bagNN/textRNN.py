# -*- coding: utf-8 -*-

"""

    x0        x1       x2
    |         |         |
    |         |         |
--> LSTM --> LSTM --> LSTM -->                       |
    |         |         |                            |
    |         |         |                            |
<-- LSTM <-- LSTM <--  LSTM                          |       bi-directional layer
    |                                                |
    |                                                |
    concat   concat    concat                        |
    |
    |
    ---------------------------------------
     K-MaxPooling
     |
     softmax

"""


class TextRNN(object):
    pass
