import numpy as np
from .rpp import RPP


class Agent:
    def __init__(self, data_loader, k, data_kind='train'):
        self.data = data_loader.data_train if data_kind == 'train' else data_loader.data_test
        self.data['rpp_action'] = 'None'
        self.prices = np.array(data_loader.data_train.open)
        min_price = np.min(self.prices)
        max_price = np.max(self.prices)

        self.rpp = RPP(k, min_price, max_price)
        self.own_share = False
        self.i_max = 1
        self.i_min = 1

    def trade(self):
        # todo: compelled to buy or sell the remaining k units in the end of the period
        k_iteration_max = self.rpp.k
        k_iteration_min = self.rpp.k
        for i in range(len(self.prices)):
            reserved_price_max = self.rpp.get_pi_max(self.i_max)
            reserved_price_min = self.rpp.get_pi_min(self.i_min)

            # print(f"Iteration {i}: RPP_max: {reserved_price_max}, RPP_min: {reserved_price_min}")

            if self.prices[i] >= reserved_price_max:
                self.i_max += 1
                k_iteration_max -= 1
                self.buy_sell('sell', i)
            elif self.prices[i] <= reserved_price_min:
                self.i_min += 1
                k_iteration_min -= 1
                self.buy_sell('buy', i)
            else:
                self.buy_sell('None', i)

            # if k_iteration_max == 0 or k_iteration_min == 0:
            #     print('Done')
            #     self.buy_sell('sell', i)
            #     break

    def buy_sell(self, action, index):
        self.data['rpp_action'][index] = action
