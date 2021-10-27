from .data import Data


class DqnData(Data):
    def __init__(self, data, action_name, device, gamma, n_step=4, batch_size=50):
        """
        This data dedicates to non-sequential models. For this, we purely pass the observation space to the agent
        by candles or some representation of the candles. We even take a window of candles as input to such models
        despite being non-time-series to see how they perform on sequential data.
        :@param state_mode
                = 1 for OHLC
                = 2 for OHLC + trend
                = 3 for OHLC + trend + %body + %upper-shadow + %lower-shadow
                = 4 for %body + %upper-shadow + %lower-shadow
                = 5 a window of k candles + the trend of the candles inside the window
        :@param action_name
            Name of the column of the action which will be added to the data-frame of data after finding the strategy by
            a specific model.
        :@param device
            GPU or CPU selected by pytorch
        @param n_step: number of steps in the future to get reward.
        @param batch_size: create batches of observations of size batch_size
        @param window_size: the number of sequential candles that are selected to be in one observation
        @param transaction_cost: cost of the transaction which is applied in the reward function.
        """
        super().__init__(data, action_name, device, gamma, n_step, batch_size, start_index_reward=0,
                         transaction_cost=0.0)

        self.data_preprocessed = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm']].values

        self.state_size = 4

        for i in range(len(self.data_preprocessed)):
            self.states.append(self.data_preprocessed[i])
