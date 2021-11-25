from data_loader import YahooFinanceDataLoader, DqnData
from evaluation import Eval
from rpp import Agent as rppAgent
from dqn import Agent as dqnAgent
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import torch
import os
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Trader arguments')
parser.add_argument('--dataset-name', default="BTC-USD",
                    help='Name of the data inside the Data folder')
parser.add_argument('--nep', type=int, default=30,
                    help='Number of episodes')
parser.add_argument('--test_type', default="train",
                    help='Evaluate the model on train or test data')
parser.add_argument('--k-range', type=int, default=100,
                    help='Max k to be tested starting from 1')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

DATA_LOADERS = {
    'BTC-USD': YahooFinanceDataLoader('BTC-USD',
                                      split_point='2018-01-01',
                                      load_from_file=True),

    'GOOGL': YahooFinanceDataLoader('GOOGL',
                                    split_point='2018-01-01',
                                    load_from_file=True),

    'AAPL': YahooFinanceDataLoader('AAPL',
                                   split_point='2018-01-01',
                                   begin_date='2010-01-01',
                                   end_date='2020-08-24',
                                   load_from_file=True),

    'DJI': YahooFinanceDataLoader('DJI',
                                  split_point='2016-01-01',
                                  begin_date='2009-01-01',
                                  end_date='2018-09-30',
                                  load_from_file=True),

    'S&P': YahooFinanceDataLoader('S&P',
                                  split_point=2000,
                                  end_date='2018-09-25',
                                  load_from_file=True),

    'AMD': YahooFinanceDataLoader('AMD',
                                  split_point=2000,
                                  end_date='2018-09-25',
                                  load_from_file=True),

    'GE': YahooFinanceDataLoader('GE',
                                 split_point='2015-01-01',
                                 load_from_file=True),

    'KSS': YahooFinanceDataLoader('KSS',
                                  split_point='2018-01-01',
                                  load_from_file=True),

    'HSI': YahooFinanceDataLoader('HSI',
                                  split_point='2015-01-01',
                                  load_from_file=True),

    'AAL': YahooFinanceDataLoader('AAL',
                                  split_point='2018-01-01',
                                  load_from_file=True)
}

experiment_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               f'Results/{args.dataset_name}/{args.test_type}')


def plot_results(portfolios, data_loader):
    plot_path = os.path.join(experiment_path, 'plots')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    plot_portfolios(data_loader, plot_path, portfolios)
    plot_sharpe_ratio(plot_path, portfolios)


def plot_sharpe_ratio(plot_path, portfolios):
    sns.set(rc={'figure.figsize': (12, 6)})
    # sns.set_palette(sns.color_palette("Paired", 15))
    plt.figure(figsize=(12, 6))
    # Calculate sharpe ratio
    sharp_ratios = {}
    for key, val in portfolios.items():
        returns = np.array(val[1:]) - np.array(val[:-1])
        sharp_ratios[key] = np.mean(returns) / np.std(returns)
    plt.bar(list(sharp_ratios.keys()), list(sharp_ratios.values()))
    plt.xlabel('Model')
    plt.ylabel('Sharpe Ratio')
    plt.title(f'Analyzing the performance of portfolios using sharpe ratio')
    fig_file = os.path.join(plot_path, args.dataset_name + '_sharpe.jpg')
    plt.savefig(fig_file, dpi=300)


def plot_portfolios(data_loader, plot_path, portfolios):
    sns.set(rc={'figure.figsize': (15, 7)})
    # sns.set_palette(sns.color_palette("Paired", 15))
    first = True
    ax = None

    for model_name in portfolios.keys():
        profit_percentage = [
            (portfolios[model_name][i] - portfolios[model_name][0]) /
            portfolios[model_name][0] * 100
            for i in range(len(portfolios[model_name]))]

        difference = len(portfolios[model_name]) - len(data_loader.data_test_with_date)
        df = pd.DataFrame({'date': data_loader.data_test_with_date.index,
                           'portfolio': profit_percentage[difference:]})
        if not first:
            df.plot(ax=ax, x='date', y='portfolio', label=model_name)
        else:
            ax = df.plot(x='date', y='portfolio', label=model_name)
            first = False
    ax.set(xlabel='Time', ylabel='%Rate of Return')
    ax.set_title(f'Analyzing the performance of portfolios')
    plt.legend()
    fig_file = os.path.join(plot_path, args.dataset_name + '_profit.jpg')
    plt.savefig(fig_file, dpi=300)


def get_buy_and_hold_portfolio(data_loader, test_type):
    data = data_loader.data_train if test_type == 'train' else data_loader.data_test
    data['buy&hold'] = 'None'
    data['buy&hold'][0] = 'buy'
    eval = Eval(data, 'buy&hold', 1000)
    return eval.get_daily_portfolio_value()


def rpp_whole_data(data, portfolios, k_max, experiment_path):
    portfolios_buy_rpp = {}
    portfolios_sell_rpp = {}
    portfolio_both_rpp = {}
    for k in tqdm(range(1, k_max)):
        try:
            agent = rppAgent(data, k, experiment_path=experiment_path)
            agent.trade()
            agent.plot_strategy_buy()
            agent.plot_strategy_sell()
            portfolios_buy_rpp[k] = agent.calculate_portfolio_buy()
            portfolios_sell_rpp[k] = agent.calculate_portfolio_sell()
            portfolio_both_rpp[k] = agent.calculate_portfolio_both()
        except AssertionError as ae:
            pass
    # find best performing buy-rpp
    best_portfolio_buy = None
    best_k_buy = None
    for k in portfolios_buy_rpp.keys():
        if best_portfolio_buy is None or portfolios_buy_rpp[k][-1] > best_portfolio_buy[-1]:
            best_portfolio_buy = portfolios_buy_rpp[k]
            best_k_buy = k
    # find best performing buy-rpp
    best_portfolio_sell = None
    best_k_sell = None
    for k in portfolios_sell_rpp.keys():
        if best_portfolio_sell is None or portfolios_sell_rpp[k][-1] > best_portfolio_sell[-1]:
            best_portfolio_sell = portfolios_sell_rpp[k]
            best_k_sell = k
    # TODO best performing both rpp
    portfolios[f'rpp_{best_k_buy}_buy'] = best_portfolio_buy
    portfolios[f'rpp_{best_k_sell}_sell'] = best_portfolio_sell


def rpp_divided_data(data, portfolios, k_max, number_of_intervals, experiment_path, initial_cash=1000):
    portfolios_buy_rpp = {}
    portfolios_sell_rpp = {}
    interval_length = int(len(data) / number_of_intervals) + 1 if len(data) % number_of_intervals != 0 \
        else int(len(data) / number_of_intervals)
    data_interval = []
    for i in range(number_of_intervals):
        data_interval.append(data[i * interval_length:(i + 1) * interval_length])

    for k in tqdm(range(1, k_max)):
        portfolios_buy_rpp[k] = []
        portfolios_sell_rpp[k] = []
        initial_investment_buy = initial_cash
        initial_investment_sell = initial_cash
        interval_index = 1
        for data_i in data_interval:
            try:
                agent = rppAgent(data_i, k, experiment_path=experiment_path, initial_cash_buy=initial_investment_buy,
                                 initial_cash_sell=initial_investment_sell)
                agent.trade()
                agent.plot_strategy_buy(file_name=f'rpp_{k}_interval_{interval_index}_buy')
                agent.plot_strategy_sell(file_name=f'rpp_{k}_interval_{interval_index}_sell')
                portfolios_buy_rpp[k].extend(agent.calculate_portfolio_buy())
                portfolios_sell_rpp[k].extend(agent.calculate_portfolio_sell())

            except AssertionError as ae:
                last_portfo_buy = portfolios_buy_rpp[k][-1] if len(
                    portfolios_buy_rpp[k]) > 0 else initial_investment_buy
                portfolios_buy_rpp[k].extend([last_portfo_buy for _ in range(len(data_i))])

                last_portfo_sell = portfolios_sell_rpp[k][-1] if len(
                    portfolios_sell_rpp[k]) > 0 else initial_investment_sell
                portfolios_sell_rpp[k].extend([last_portfo_sell for _ in range(len(data_i))])

            initial_investment_buy = portfolios_buy_rpp[k][-1]
            initial_investment_sell = portfolios_sell_rpp[k][-1]
            interval_index += 1
    # find best performing buy-rpp
    best_portfolio_buy = None
    best_k_buy = None
    for k in portfolios_buy_rpp.keys():
        if best_portfolio_buy is None or portfolios_buy_rpp[k][-1] > best_portfolio_buy[-1]:
            best_portfolio_buy = portfolios_buy_rpp[k]
            best_k_buy = k
    # find best performing buy-rpp
    best_portfolio_sell = None
    best_k_sell = None
    for k in portfolios_sell_rpp.keys():
        if best_portfolio_sell is None or portfolios_sell_rpp[k][-1] > best_portfolio_sell[-1]:
            best_portfolio_sell = portfolios_sell_rpp[k]
            best_k_sell = k
    portfolios[f'rpp_interval_{best_k_buy}_buy'] = best_portfolio_buy
    portfolios[f'rpp_interval_{best_k_sell}_sell'] = best_portfolio_sell


if __name__ == '__main__':
    test_type = args.test_type
    n_step = 8
    dataset_name = args.dataset_name
    k_max = args.k_range
    n_episodes = args.nep
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    feature_size = 64
    target_update = 5
    gamma = 0.9
    batch_size = 16
    replay_memory_size = 32

    number_of_intervals = 10

    data_loader = DATA_LOADERS[dataset_name]

    # data_loader.plot_data()

    portfolios = {}

    data = data_loader.data_train if test_type == 'train' else data_loader.data_test

    if test_type == 'test':
        rpp_whole_data(data, portfolios, k_max, experiment_path)
        rpp_divided_data(data, portfolios, k_max, number_of_intervals, experiment_path)

    data_train_dqn = \
        DqnData(data=data_loader.data_train,
                action_name='action_dqn',
                device=device,
                gamma=gamma,
                n_step=n_step,
                batch_size=batch_size)

    data_test_dqn = \
        DqnData(data=data_loader.data_test,
                action_name='action_dqn',
                device=device,
                gamma=gamma,
                n_step=n_step,
                batch_size=batch_size)

    dqn = dqnAgent(data_loader,
                   data_train_dqn,
                   data_test_dqn,
                   dataset_name,
                   device,
                   BATCH_SIZE=batch_size,
                   GAMMA=gamma,
                   ReplayMemorySize=replay_memory_size,
                   TARGET_UPDATE=target_update,
                   n_step=n_step)

    if test_type == 'train':
        dqn.train(n_episodes)
    dqn_eval = dqn.test(test_type=test_type)
    portfolios['dqn'] = dqn_eval.get_daily_portfolio_value()

    dqn_eval.plot_strategy(experiment_path)

    # add buy & hold agent
    portfolios['buy&hold'] = get_buy_and_hold_portfolio(data_loader, test_type)

    plot_results(portfolios, data_loader)
