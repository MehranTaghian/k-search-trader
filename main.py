from DataLoader.DataLoader import YahooFinanceDataLoader
from Evaluation.Evaluation import Evaluation
from rpp import RPP
import numpy as np


class Agent:
    def __init__(self, data_loader, k, data_kind='train'):
        self.data = data_loader.data_train if data_kind == 'train' else data_loader.data_test
        self.data['action'] = 'None'
        self.prices = np.array(data_loader.data_train.open)
        min_price = np.min(self.prices)
        max_price = np.max(self.prices)
        self.rpp = RPP(k, min_price, max_price)
        self.own_share = False
        self.i_max = 1
        self.i_min = 1

    def trade(self):
        for i in range(len(self.prices)):
            reserved_price_max = self.rpp.get_pi_max(self.i_max)
            reserved_price_min = self.rpp.get_pi_min(self.i_min)

            if self.prices[i] >= reserved_price_max:
                self.i_max += 1
                self.buy_sell('sell', i)
            if self.prices[i] <= reserved_price_min:
                self.i_min += 1
                self.buy_sell('buy', i)

    def buy_sell(self, action, index):
        if action == 'buy' and not self.own_share:
            self.data['action'][index] = action
            self.own_share = True
        elif action == 'sell' and self.own_share:
            self.data['action'][index] = action
            self.own_share = False
        else:
            self.data['action'][index] = 'None'


if __name__ == '__main__':
    data_loader = YahooFinanceDataLoader('BTC-USD', split_point='2018-01-01', load_from_file=True)
    k = 100000
    agent = Agent(data_loader, k, data_kind='test')
    agent.trade()
    eval = Evaluation(agent.data, 'action', initial_investment=1000)
    eval.evaluate()
