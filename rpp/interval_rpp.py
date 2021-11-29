import numpy as np
from .rpp import RPP
import matplotlib.pyplot as plt
import os
import seaborn as sns


class Agent:
    def __init__(self, data, k, num_intervals, experiment_path, initial_cash_buy=1000, initial_cash_sell=1000,
                 initial_cash_both=1000):
        self.data = data
        self.k = k
        self.action_col = 'rpp_interval_action'
        self.data[self.action_col] = 'None'
        self.prices = np.array(self.data.close)
        # Dividing data into some number of intervals
        self.len_interval = int(self.prices.shape[0] / num_intervals) + 1 if self.prices.shape[0] % num_intervals != 0 \
            else int(self.prices.shape[0] / num_intervals)

        self.prices_interval = []
        for i in range(num_intervals):
            self.prices_interval.append(self.prices[i * self.len_interval:(i + 1) * self.len_interval])

        self.buy_indices = np.zeros(self.prices.shape[0])
        self.sell_indices = np.zeros(self.prices.shape[0])

        self.name = 'interval'
        self.experiment_path = os.path.join(experiment_path, 'rpp', self.name)

        if not os.path.exists(self.experiment_path):
            os.makedirs(self.experiment_path)

        # assume buying at the beginning to sell gradually
        self.num_shares_sell = initial_cash_sell / self.prices[0]

        # assume not buying anything to buy gradually
        self.initial_cash_buy = initial_cash_buy

        # When considering both buy and sell signals
        self.initial_cash_both = initial_cash_both
        self.num_shares_both = initial_cash_both / self.prices[0]

        if not os.path.exists(self.experiment_path):
            os.makedirs(self.experiment_path)

    def trade(self):
        """
        This function returns True if it was able to trade anything, else it returns False
        """
        # todo: compelled to buy or sell the remaining k units in the end of the period
        for interval in range(len(self.prices_interval)):
            prices = self.prices_interval[interval]
            min_price = np.min(prices)
            max_price = np.max(prices)
            try:
                rpp = RPP(self.k, min_price, max_price)
            except AssertionError:
                continue
            i_max = 1
            i_min = 1
            k_iteration_max = self.k
            k_iteration_min = self.k

            for index in range(prices.shape[0]):
                reserved_price_max = rpp.get_pi_max(i_max)
                reserved_price_min = rpp.get_pi_min(i_min)

                # print(f"Iteration {i}: RPP_max: {reserved_price_max}, RPP_min: {reserved_price_min}")

                if prices[index] >= reserved_price_max and k_iteration_max > 0:
                    i_max += 1
                    k_iteration_max -= 1
                    self.buy_sell('sell', interval, index)
                    self.sell_indices[interval * self.len_interval + index] = 1
                elif prices[index] <= reserved_price_min and k_iteration_min > 0:
                    i_min += 1
                    k_iteration_min -= 1
                    self.buy_sell('buy', interval, index)
                    self.buy_indices[interval * self.len_interval + index] = 1
                else:
                    self.buy_sell('None', interval, index)

                if k_iteration_max == 0 and k_iteration_min == 0:
                    self.buy_sell('sell', interval, index)
                    break

        if (self.data[self.action_col] == 'None').all():
            return False

        return True

    def buy_sell(self, action, interval_index, index):
        self.data[self.action_col][interval_index * self.len_interval + index] = action

    def calculate_portfolio_sell(self):
        unit_share = self.num_shares_sell / self.k
        portfolio = [self.prices[0] * self.num_shares_sell]
        num_shares = self.num_shares_sell.copy()
        current_cash = 0

        for i in range(1, len(self.prices)):
            if self.data[self.action_col].iloc[i] == 'sell':
                current_cash += unit_share * self.prices[i]
                num_shares -= unit_share

            portfolio.append(current_cash + self.prices[i] * num_shares)

        return portfolio

    def calculate_portfolio_buy(self):
        unit_asset = self.initial_cash_buy / self.k
        portfolio = [self.initial_cash_buy]
        current_cash = self.initial_cash_buy
        num_shares = 0
        for i in range(1, self.prices.shape[0]):
            if self.data[self.action_col].iloc[i] == 'buy':
                current_cash -= unit_asset
                num_shares += unit_asset / self.prices[i]

            portfolio.append(current_cash + self.prices[i] * num_shares)

        return portfolio

    def calculate_portfolio_both(self):
        unit_asset = self.initial_cash_both / self.k
        unit_share = self.num_shares_both.copy()
        portfolio = [self.initial_cash_both]
        current_cash = self.initial_cash_both
        num_shares = 0
        for i in range(1, self.prices.shape[0]):
            if self.data[self.action_col].iloc[i] == 'buy':
                current_cash -= unit_asset
                num_shares += unit_asset / self.prices[i]
            elif self.data[self.action_col].iloc[i] == 'sell' and num_shares >= unit_share:
                current_cash += unit_share * self.prices[i]
                num_shares -= unit_share

            portfolio.append(current_cash + self.prices[i] * num_shares)

        return portfolio

    def plot_strategy_buy(self):
        file_name = f'rpp_{self.k}_buy.jpg'
        sns.set(rc={'figure.figsize': (15, 7)})
        # sns.set_palette(sns.color_palette("Paired", 15))
        plt.figure(figsize=(15, 7))
        x_buy = np.arange(len(self.prices)) * self.buy_indices
        y_buy = self.prices * self.buy_indices

        plt.plot(self.prices, color='b', alpha=0.2)
        plt.scatter(x_buy[y_buy != 0], y_buy[y_buy != 0], color='g', label='buy')
        plt.xlabel('Day')
        plt.ylabel('Price')
        plt.title('Buy signals produced by RPP algorithm')
        plt.legend()
        plt.savefig(os.path.join(self.experiment_path, file_name), dpi=300)

    def plot_strategy_sell(self):
        file_name = f'rpp_{self.k}_sell.jpg'
        sns.set(rc={'figure.figsize': (15, 7)})
        # sns.set_palette(sns.color_palette("Paired", 15))
        plt.figure(figsize=(15, 7))
        x_sell = np.arange(len(self.prices)) * self.sell_indices
        y_sell = self.prices * self.sell_indices

        plt.plot(self.prices, color='b', alpha=0.2)
        plt.scatter(x_sell[y_sell != 0], y_sell[y_sell != 0], color='r', label='sell')
        plt.xlabel('Day')
        plt.ylabel('Price')
        plt.title('Sell signals produced by RPP algorithm')
        plt.legend()
        plt.savefig(os.path.join(self.experiment_path, file_name), dpi=300)

    def plot_strategy_both(self):
        file_name = f'rpp_{self.k}_both.jpg'
        sns.set(rc={'figure.figsize': (15, 7)})
        # sns.set_palette(sns.color_palette("Paired", 15))
        plt.figure(figsize=(15, 7))
        x_buy = np.arange(len(self.prices)) * self.buy_indices
        y_buy = self.prices * self.buy_indices
        x_sell = np.arange(len(self.prices)) * self.sell_indices
        y_sell = self.prices * self.sell_indices

        plt.plot(self.prices, color='b', alpha=0.2)
        plt.scatter(x_buy[y_buy != 0], y_buy[y_buy != 0], color='g', label='buy')
        plt.scatter(x_sell[y_sell != 0], y_sell[y_sell != 0], color='r', label='sell')
        plt.xlabel('Day')
        plt.ylabel('Price')
        plt.title('Buy and Sell signals produced by RPP algorithm')
        plt.legend()
        plt.savefig(os.path.join(self.experiment_path, file_name), dpi=300)
