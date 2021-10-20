from rpp import RPP
import numpy as np


class Agent:
    def __init__(self, data_loader, k):
        prices = np.array(data_loader.data_train.close)
        min_price = np.min(prices)
        max_price = np.max(prices)
        self.rpp = RPP(k, min_price, max_price)
        self.own_share = False



