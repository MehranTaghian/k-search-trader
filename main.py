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
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Trader arguments')
parser.add_argument('--dataset-name', default="BTC-USD",
                    help='Name of the data inside the Data folder')
parser.add_argument('--nep', type=int, default=30,
                    help='Number of episodes')
parser.add_argument('--test_type', default="train",
                    help='Evaluate the model on train or test data')
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

    sns.set(rc={'figure.figsize': (15, 7)})
    sns.set_palette(sns.color_palette("Paired", 15))

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
    fig_file = os.path.join(plot_path, args.dataset_name + '.jpg')
    plt.savefig(fig_file, dpi=300)


if __name__ == '__main__':
    test_type = args.test_type
    n_step = 8
    dataset_name = args.dataset_name
    n_episodes = args.nep
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    feature_size = 64
    target_update = 5

    gamma = 0.9
    batch_size = 16
    replay_memory_size_default = 32

    data_loader = DATA_LOADERS[dataset_name]

    portfolios = {}

    k = 10
    for k in tqdm(range(1, 50)):
        try:
            agent = rppAgent(data_loader, k, experiment_path=experiment_path, data_kind=test_type)
            agent.trade()
            agent.plot_strategy()
            portfolios['rpp_sell_' + str(k)] = agent.calculate_portfolio_sell()
            portfolios['rpp_buy_' + str(k)] = agent.calculate_portfolio_buy()
            print(k, portfolios['rpp_sell_' + str(k)][-1])
            print(k, portfolios['rpp_buy_' + str(k)][-1])
        except AssertionError as ae:
            pass

    # data_train_dqn = \
    #     DqnData(data=data_loader.data_train,
    #             action_name='action_dqn',
    #             device=device,
    #             gamma=gamma,
    #             n_step=n_step,
    #             batch_size=batch_size)
    #
    # data_test_dqn = \
    #     DqnData(data=data_loader.data_test,
    #             action_name='action_dqn',
    #             device=device,
    #             gamma=gamma,
    #             n_step=n_step,
    #             batch_size=batch_size)
    #
    # dqn = dqnAgent(data_loader,
    #                data_train_dqn,
    #                data_test_dqn,
    #                dataset_name,
    #                BATCH_SIZE=30,
    #                GAMMA=0.7,
    #                ReplayMemorySize=50,
    #                TARGET_UPDATE=5,
    #                n_step=10)
    #
    # dqn.train(n_episodes)
    # dqn_eval = dqn.test(test_type=test_type)
    # portfolios['dqn'] = dqn_eval.get_daily_portfolio_value()

    plot_results(portfolios, data_loader)
