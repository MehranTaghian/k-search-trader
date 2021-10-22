from DataLoader.DataLoader import YahooFinanceDataLoader
from Evaluation.Evaluation import Evaluation
from rpp import RPP
import numpy as np


class Agent:
    def __init__(self, data_loader, k, data_kind='train'):
        self.data = data_loader.data_train if data_kind == 'train' else data_loader.data_test
        self.data['action'] = 'None'
        self.prices = np.array(data_loader.data_train.close)
        min_price = np.min(self.prices)
        max_price = np.max(self.prices)

        self.rpp = RPP(k, min_price, max_price)
        self.own_share = False
        self.i_max = 1
        self.i_min = 1

    def trade(self):
        k_iteration = self.rpp.k
        for i in range(len(self.prices)):
            reserved_price_max = self.rpp.get_pi_max(self.i_max)
            reserved_price_min = self.rpp.get_pi_min(self.i_min)

            # print(f"Iteration {i}: RPP_max: {reserved_price_max}, RPP_min: {reserved_price_min}")

            # print(self.prices[i])
            if self.prices[i] >= reserved_price_max:
                self.i_max += 1
                # print("yes bigger")
                self.buy_sell('sell', i)
                k_iteration -= 1
            elif self.prices[i] <= reserved_price_min:
                self.i_min += 1
                self.buy_sell('buy', i)
                k_iteration -= 1
            else:
                self.buy_sell('None', i)

            if k_iteration == 0:
                break

    def buy_sell(self, action, index):
        self.data['action'][index] = action
        # if action == 'buy' and not self.own_share:
        #     self.data['action'][index] = action
        #     self.own_share = True
        # elif action == 'sell' and self.own_share:
        #     self.data['action'][index] = action
        #     self.own_share = False
        # else:
        #     self.data['action'][index] = 'None'


if __name__ == '__main__':
    data_loader = YahooFinanceDataLoader('BTC-USD', split_point='2018-01-01', load_from_file=True)
    k_list = [10, 100, 1000, 10000]
    for k in range(500):
        try:
            agent = Agent(data_loader, k, data_kind='test')
            agent.trade()
            eval = Evaluation(agent.data, 'action', initial_investment=1000)
            print(eval.get_daily_portfolio_value()[-1])
        except AssertionError as ae:
            print(ae)