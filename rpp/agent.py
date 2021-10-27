import numpy as np
from .rpp import RPP
import matplotlib.pyplot as plt
import os
import seaborn as sns


class Agent:
    def __init__(self, data_loader, k, experiment_path, data_kind='train', initial_cash=1000):
        self.data = data_loader.data_train if data_kind == 'train' else data_loader.data_test
        self.data['rpp_action'] = 'None'
        self.prices = np.array(self.data.close)
        self.buy_indices = np.zeros(self.prices.shape[0])
        self.sell_indices = np.zeros(self.prices.shape[0])

        min_price = np.min(self.prices)
        max_price = np.max(self.prices)
        self.rpp = RPP(k, min_price, max_price)
        self.own_share = False
        self.i_max = 1
        self.i_min = 1

        self.experiment_path = os.path.join(experiment_path, 'rpp')

        # assume buying at the beginning to sell gradually
        self.num_shares_sell = initial_cash / self.prices[0]

        # assume not buying anything to buy gradually
        self.initial_cash = initial_cash

        if not os.path.exists(self.experiment_path):
            os.makedirs(self.experiment_path)

    def trade(self):
        # todo: compelled to buy or sell the remaining k units in the end of the period
        k_iteration_max = self.rpp.k
        k_iteration_min = self.rpp.k
        for i in range(len(self.prices)):
            reserved_price_max = self.rpp.get_pi_max(self.i_max)
            reserved_price_min = self.rpp.get_pi_min(self.i_min)

            # print(f"Iteration {i}: RPP_max: {reserved_price_max}, RPP_min: {reserved_price_min}")

            if self.prices[i] >= reserved_price_max and k_iteration_max > 0:
                self.i_max += 1
                k_iteration_max -= 1
                self.buy_sell('sell', i)
                self.sell_indices[i] = 1
            elif self.prices[i] <= reserved_price_min and k_iteration_min > 0:
                self.i_min += 1
                k_iteration_min -= 1
                self.buy_sell('buy', i)
                self.buy_indices[i] = 1
            else:
                self.buy_sell('None', i)

            if k_iteration_max == 0 and k_iteration_min == 0:
                self.buy_sell('sell', i)
                break

    def buy_sell(self, action, index):
        self.data['rpp_action'][index] = action

    def calculate_portfolio_sell(self):
        unit_share = self.num_shares_sell / self.rpp.k
        portfolio = [self.prices[0] * self.num_shares_sell]
        num_shares = self.num_shares_sell.copy()
        current_cash = 0

        for i in range(1, len(self.prices)):
            if self.data['rpp_action'][i] == 'sell':
                current_cash += unit_share * self.prices[i]
                num_shares -= unit_share

            portfolio.append(current_cash + self.prices[i] * num_shares)

        return portfolio

    def calculate_portfolio_buy(self):
        unit_asset = self.initial_cash / self.rpp.k
        portfolio = [self.initial_cash]
        current_cash = self.initial_cash
        num_shares = 0

        print(current_cash)

        for i in range(1, len(self.prices)):
            if self.data['rpp_action'][i] == 'buy':
                current_cash -= unit_asset
                num_shares += unit_asset / self.prices[i]

            portfolio.append(current_cash + self.prices[i] * num_shares)

        return portfolio

    def plot_strategy(self):
        sns.set(rc={'figure.figsize': (15, 7)})
        sns.set_palette(sns.color_palette("Paired", 15))
        x_buy = np.arange(len(self.prices)) * self.buy_indices
        x_sell = np.arange(len(self.prices)) * self.sell_indices
        y_buy = self.prices * self.buy_indices
        y_sell = self.prices * self.sell_indices

        plt.plot(self.prices, color='b', alpha=0.2)
        plt.scatter(x_buy[y_buy != 0], y_buy[y_buy != 0], color='g')
        plt.scatter(x_sell[y_sell != 0], y_sell[y_sell != 0], color='r')
        plt.savefig(self.experiment_path + f'/rpp_{self.rpp.k}.jpg', dpi=300)
