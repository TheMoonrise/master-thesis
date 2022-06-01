import math

import numpy as np

import json
import gym
from typing import Tuple, Dict, List, Union
import importlib.resources
import json
import pandas as pd
import random
import warnings


class DataSource:
    FIVE_MIN_AGGREGATE = "5min_aggregate"
    PER_BLOCK = "per_block"


class CryptoMarketsEnv(gym.Env):

    LIST_STATE_RELEVANT_ASSET_COLUMNS = [
        "vol_token0",
        "vol_token1",
        "tvl_token1",
        "tx",
        "max_to_close",
        "min_to_close",
        "close_to_close",
    ]

    LIST_STATE_RELEVANT_FEE_COLUMNS = ["avg_gas_price_per_block_gwei"]

    DICT_DATA_SOURCE = {
        DataSource.FIVE_MIN_AGGREGATE: {
            "fee_data": "crypto_fee_data_total",
            "pool_data": "crypto_pool_data_total"
        },
        DataSource.PER_BLOCK: {
            "fee_data": "crypto_fee_data_single_block_total",
            "pool_data": "crypto_pool_data_single_block_total"
        }
    }

    def __init__(
        self,
        config,
        **kwargs,
    ):

        # NOTE input parameters replaced by config
        # parameters are now fetched from the config below
        amount_time_steps_to_include = (12 * 24)
        starting_funds = 1.0
        fee_difficulty_scaling_factor = 1.0
        data_source = DataSource.FIVE_MIN_AGGREGATE
        test_data_usage_percentage = 0.2
        include_gas_bid_in_action = False

        self.meta_actions = False
        self.softmax_actions = False

        if 'starting_funds' in config: starting_funds = config['starting_funds']
        if 'fee_difficulty_scaling_factor' in config: fee_difficulty_scaling_factor = config['fee_difficulty_scaling_factor']
        if 'test_data_usage_percentage' in config: test_data_usage_percentage = config['test_data_usage_percentage']

        if 'softmax_actions' in config: self.softmax_actions = config['softmax_actions']
        if 'meta_actions' in config: self.meta_actions = config['meta_actions']

        if 'is_validation' in config: self.set_test_data_mode(config['is_validation'])

        print(f"STARTING FUNDS {starting_funds}")

        self.amount_pairs_to_include = 10 + 1

        self.data_source = data_source

        self.list_relevant_pairs = self.initialize_list_relevant_pairs()
        self.total_crypto_data = self.initializing_crypto_data()
        self.total_crypto_fee_data = self.initializing_crypto_fee_data()
        self.total_amount_observations_per_asset = self.total_crypto_data["index"].max()

        self.total_amount_observations_per_asset_training_data = \
            math.floor(self.total_amount_observations_per_asset * (1 - test_data_usage_percentage))

        self.fee_difficulty_scaling_factor = fee_difficulty_scaling_factor
        self.include_gas_bid_in_action = include_gas_bid_in_action

        # Optional, allows for utilization of a testing dataset, if set true the environment only samples
        # from the test data set, identified by index
        self.is_test_data_mode = False

        #
        self.is_bankrupt = False

        # starting amount of ETH
        self.starting_funds = starting_funds
        self.current_state_funds = self.starting_funds
        self.current_state_portfolio_allocation = self.generate_allocation_encoding(0)

        self.amount_time_steps_to_include = amount_time_steps_to_include
        self.current_initial_time_step = self.sample_allowed_initial_time_index()
        self.current_trajectory_time_step_counter = 0

        self.current_trajectory_df = self.build_current_trajectory_data(
            initial_time_index=self.current_initial_time_step
        )

        self.current_trajectory_fee_df = self.build_current_trajectory_fee_data(
            initial_time_index=self.current_initial_time_step
        )

        if not self.include_gas_bid_in_action:
            self.action_space = gym.spaces.Discrete(self.amount_pairs_to_include)
        else:
            spaces = {
                'gas_price_bid': gym.spaces.Box(low=0, high=1000, shape=(1,)),
                # in GWEI - max is set to 1000
                'investment_decision': gym.spaces.Discrete(self.amount_pairs_to_include)
            }
            self.action_space = gym.spaces.Dict(spaces)

        observation_space_size = (
            1 + self.amount_pairs_to_include + 1 + self.current_trajectory_df.shape[1]
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_space_size,), dtype=float
        )

    @staticmethod
    def check_all_elements_included(
        np_observation: np.ndarray, last_read_element_index: int
    ):
        if np_observation.ndim == 2:
            assert np_observation.shape[1] == last_read_element_index
        elif np_observation.ndim == 1:
            assert np_observation.shape[0] == last_read_element_index

    @staticmethod
    def calculate_amount_assets(np_observation: np.ndarray) -> int:
        additional_fields = 0
        # portfolio funds
        additional_fields += 1
        # latest average in gwei per gas in previous block
        additional_fields += len(CryptoMarketsEnv.LIST_STATE_RELEVANT_FEE_COLUMNS)
        return int(
            (np_observation.shape[0] - additional_fields)
            / (len(CryptoMarketsEnv.LIST_STATE_RELEVANT_ASSET_COLUMNS) + 1)
        )

    @staticmethod
    def static_decompose_environment_observation_dict(
        np_observation: np.ndarray,
    ) -> Tuple:

        # crypto_funds #in ETH
        # current_state_portfolio_allocation
        # latest average in gwei per gas in the previous block
        # latest observations from previous block of LIST_STATE_RELEVANT_ASSET_COLUMNS per asset

        if np_observation.ndim == 2:
            amount_assets = CryptoMarketsEnv.calculate_amount_assets(
                np_observation[0, :].flatten()
            )
            crypto_funds = np_observation[:, 0]
            last_observed_avg_gas_fee = np_observation[:, 1]
            current_state_portfolio_allocation = np_observation[
                :, 2: 2 + amount_assets
            ]
            final_index_element = (
                2
                + amount_assets
                + len(CryptoMarketsEnv.LIST_STATE_RELEVANT_ASSET_COLUMNS)
                * amount_assets
            )
            last_observed_market_data = np_observation[
                :, (2 + amount_assets): final_index_element
            ]
            CryptoMarketsEnv.check_all_elements_included(
                np_observation, final_index_element
            )
            tmp_dict = {
                "portfolio_funds": crypto_funds,
                "last_observed_avg_gas_fee": last_observed_avg_gas_fee,
                "current_state_portfolio_allocation": current_state_portfolio_allocation,
                "last_observed_market_data": last_observed_market_data,
            }
            return tmp_dict
        elif np_observation.ndim == 1:
            amount_assets = CryptoMarketsEnv.calculate_amount_assets(np_observation)
            crypto_funds = np_observation[0]
            last_observed_avg_gas_fee = np_observation[1]
            current_state_portfolio_allocation = np_observation[2: 2 + amount_assets]
            final_index_element = (
                2
                + amount_assets
                + len(CryptoMarketsEnv.LIST_STATE_RELEVANT_ASSET_COLUMNS)
                * amount_assets
            )
            last_observed_market_data = np_observation[
                (2 + amount_assets): final_index_element
            ]
            CryptoMarketsEnv.check_all_elements_included(
                np_observation, final_index_element
            )
            tmp_dict = {
                "portfolio_funds": crypto_funds,
                "last_observed_avg_gas_fee": last_observed_avg_gas_fee,
                "current_state_portfolio_allocation": current_state_portfolio_allocation,
                "last_observed_market_data": last_observed_market_data,
            }
            return tmp_dict

    def generate_allocation_encoding(self, asset_index: int) -> np.ndarray:
        np_allocation_encoding = np.zeros(self.amount_pairs_to_include, dtype=float)
        np_allocation_encoding[asset_index] = 1.0
        return np_allocation_encoding

    def generate_allocation_decoding(self, np_allocation_encoding: np.ndarray) -> int:
        return np.argmax(np_allocation_encoding).item()

    def sample_allowed_initial_time_index(self) -> int:
        if not self.is_test_data_mode:
            return random.randint(
                0,
                (
                    self.total_amount_observations_per_asset_training_data
                    - self.amount_time_steps_to_include
                    - 1
                ),
            )  # -1 due to 0 indexing
        elif self.is_test_data_mode:
            assert (
                self.total_amount_observations_per_asset
                - self.amount_time_steps_to_include
                - 1
            ) > (self.total_amount_observations_per_asset_training_data + 1), f"Test data set must be bigger," \
                f" consider changing 'test_data_usage_percentage'"
            return random.randint(
                self.total_amount_observations_per_asset_training_data + 1,
                (
                    self.total_amount_observations_per_asset
                    - self.amount_time_steps_to_include
                    - 1
                ),
            )  # -1 due to 0 indexing

    def initializing_crypto_fee_data(self) -> pd.DataFrame:

        # https://stackoverflow.com/questions/779495/access-data-in-package-subdirectory
        with importlib.resources.open_binary(
            "crypto_markets_gym.envs.data",
                f"{CryptoMarketsEnv.DICT_DATA_SOURCE.get(self.data_source).get('fee_data')}.ftr"
        ) as file:
            columns_to_read = [
                "index",
                # "associated_time_slot",
                "avg_gas_price_per_block_gwei",
            ]

            crypto_fee_df = pd.read_feather(file, columns=columns_to_read)

        return crypto_fee_df

    def initializing_crypto_data(self) -> pd.DataFrame:

        # https://stackoverflow.com/questions/779495/access-data-in-package-subdirectory
        with importlib.resources.open_binary(
            "crypto_markets_gym.envs.data",
                f"{CryptoMarketsEnv.DICT_DATA_SOURCE.get(self.data_source).get('pool_data')}.ftr"
        ) as file:
            columns_to_read = [
                "index",
                # "associated_time_slot",
                "token0",
                "vol_token0",
                "token1",
                "vol_token1",
                "tvl_token1",
                "tx",
                "trading_pair",
                "max_to_close",
                "min_to_close",
                "close_to_close",
            ]

            crypto_df = pd.read_feather(file, columns=columns_to_read)
            # transforming data
            tmp_list_dfs = []
            for key_pair in [
                f'{entry.get("token0").get("symbol")}_{entry.get("token1").get("symbol")}'
                for entry in self.list_relevant_pairs
            ]:
                tmp_df = crypto_df[
                    crypto_df["trading_pair"] == key_pair
                ]

                tmp_df = tmp_df.set_index("index")
                tmp_df = tmp_df[CryptoMarketsEnv.LIST_STATE_RELEVANT_ASSET_COLUMNS]
                tmp_df.columns = [f"{key_pair}_{tmp_entry}" for tmp_entry in tmp_df.columns]
                tmp_list_dfs.append(tmp_df)

            crypto_df = pd.concat(tmp_list_dfs, axis=1)
            crypto_df["index"] = crypto_df.index
            # end transformation
        return crypto_df

    def initialize_list_relevant_pairs(self) -> List[Dict]:

        pair_file = "uni_v3_pairs.json"

        with importlib.resources.open_text(
            "crypto_markets_gym.envs.data", f"{pair_file}"
        ) as file:
            list_pairs = json.load(file)

        #tmp_list_unique = self.total_crypto_data["trading_pair"].unique()

        list_relevant_pairs = [
            val
            for ind, val in enumerate(list_pairs)
            # if (
            #    f'{val.get("token0").get("symbol")}_{val.get("token1").get("symbol")}'
            #    in tmp_list_unique
            # )
            # &
            if (ind < self.amount_pairs_to_include)
        ]
        return list_relevant_pairs

    def evaluate_trading_and_fee(
        self,
        current_state_funds,
        previous_state_portfolio_allocation,
        current_state_portfolio_allocation,
        current_trajectory_time_step_counter,
        agent_gas_price_gwei: float = None,
    ) -> Tuple[float, bool]:

        # we assume a gas cost of 160_000
        # (160_000/1_000_000_000)*GWEI_PRICE #assuming that we have 160_000 gas cost for an Asset1 -> ETH -> Asset2 trip
        trading_delta_vector = (
            current_state_portfolio_allocation - previous_state_portfolio_allocation
        )
        previous_idx_asset = np.argmax(previous_state_portfolio_allocation)
        current_idx_asset = np.argmax(current_state_portfolio_allocation)

        if np.sum(np.abs(trading_delta_vector)) > 0:
            # 1) pool fee
            previous_trading_pair_trading_fee = (
                float(self.list_relevant_pairs[previous_idx_asset].get("feeTier"))
                / 1000000.0
            )  # measured in 1/millionth
            current_trading_pair_trading_fee = (
                float(self.list_relevant_pairs[current_idx_asset].get("feeTier"))
                / 1000000.0
            )  # measured in 1/millionth
            # We charge as gas/gwei the price in the previous observed 5min window
            # 2) ETH gas fee
            if self.include_gas_bid_in_action:
                current_gas_price_ETH = agent_gas_price_gwei / 1.0e+09
            else:
                current_gas_price_ETH = float(
                    self.current_trajectory_fee_df.to_numpy()[
                        current_trajectory_time_step_counter, :
                    ]
                    / 1.0e+09  # 1_000_000_000
                )  # measured in gwei

            fixed_network_fee_swap = (
                160_000 * current_gas_price_ETH
            )  # assumption of 160_000 gas required per swap

            # 3) Price Impact
            str_trading_pair = self.get_str_trading_pair(current_idx_asset)
            current_price_impact_relative = (current_state_funds *
                                             (1 - previous_trading_pair_trading_fee -
                                              current_trading_pair_trading_fee)) / float(
                self.current_trajectory_df.iloc[
                    current_trajectory_time_step_counter
                ][f"{str_trading_pair}_tvl_token1"]
            )  # price impact is (amount token A pool in)/(amount token A pool locked)

            # -> The return impact will be 1-1/(1+price_impact), i.e. a +200% price impact (you paid three times
            # the current fair market value) will reduce your funds value to 33%, i.e. we paid a 66% "price impact fee"
            return_impact_current_price_impact = 1.0 - (1.0 / (1.0 + current_price_impact_relative))

            total_fee_relative = fixed_network_fee_swap / current_state_funds + (
                previous_trading_pair_trading_fee + current_trading_pair_trading_fee
            ) + return_impact_current_price_impact
            difficulty_adjusted_fee_relative = total_fee_relative * self.fee_difficulty_scaling_factor

            trade_occured = True
            return difficulty_adjusted_fee_relative, trade_occured
        else:
            trade_occured = False
            return 0.0, trade_occured

    def set_fee_difficulty_scaling_factor(self, fee_difficulty_scaling_factor: float):
        self.fee_difficulty_scaling_factor = fee_difficulty_scaling_factor

    def build_current_trajectory_data(self, initial_time_index: int) -> pd.DataFrame:
        tmp_df = self.total_crypto_data

        tmp_df = tmp_df[
            (tmp_df["index"] >= initial_time_index)
            & (
                tmp_df["index"]
                < (initial_time_index + self.amount_time_steps_to_include)
            )
        ]
        tmp_df = tmp_df.set_index("index")

        return tmp_df

    def build_current_trajectory_fee_data(
        self, initial_time_index: int
    ) -> pd.DataFrame:
        tmp_df = self.total_crypto_fee_data

        tmp_df = tmp_df[
            (tmp_df["index"] >= initial_time_index)
            & (
                tmp_df["index"]
                < (initial_time_index + self.amount_time_steps_to_include)
            )
        ]
        tmp_df = tmp_df.set_index("index")
        tmp_df = tmp_df[CryptoMarketsEnv.LIST_STATE_RELEVANT_FEE_COLUMNS]

        return tmp_df

    def render(self):
        pass

    def set_test_data_mode(self, is_test_data_mode=False):
        self.is_test_data_mode = is_test_data_mode

    def get_str_trading_pair(self, idx):
        str_trading_pair = (
            f'{self.list_relevant_pairs[idx].get("token0").get("symbol")}_'
            f'{self.list_relevant_pairs[idx].get("token1").get("symbol")}'
        )
        return str_trading_pair

    def is_larger_than_upcoming_network_gas_price(self, agent_gas_price_bid, current_trajectory_time_step_counter):
        upcoming_network_gas_price = float(
            self.current_trajectory_fee_df.to_numpy()[(current_trajectory_time_step_counter + 1), :])
        return agent_gas_price_bid >= upcoming_network_gas_price

    def step(self, action: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, float, bool, dict]:

        # NOTE if meta actions are enabled the action is sampled here
        if self.softmax_actions: action = np.exp(action) / np.sum(np.exp(action), axis=0)
        if self.meta_actions: action = np.array(np.random.choice(self.amount_pairs_to_include, p=action))

        if self.include_gas_bid_in_action:
            idx_action_asset_to_invest = action.get("investment_decision").item()
            agent_gas_price_bid = action.get("gas_price_bid").item()
        else:
            agent_gas_price_bid = None
            idx_action_asset_to_invest = (
                action.item()
            )

        # Check for trade eligibility
        if self.current_state_funds > 0.0:
            if self.include_gas_bid_in_action:
                trade_eligible = self.is_larger_than_upcoming_network_gas_price(
                    agent_gas_price_bid=agent_gas_price_bid,
                    current_trajectory_time_step_counter=self.current_trajectory_time_step_counter)
            else:
                trade_eligible = True
        else:
            trade_eligible = False

        previous_state_portfolio_allocation = self.current_state_portfolio_allocation

        if trade_eligible:
            self.current_state_portfolio_allocation = self.generate_allocation_encoding(
                idx_action_asset_to_invest
            )
            current_fee, trade_occured = self.evaluate_trading_and_fee(
                current_state_funds=self.current_state_funds,
                previous_state_portfolio_allocation=previous_state_portfolio_allocation,
                current_state_portfolio_allocation=self.current_state_portfolio_allocation,
                current_trajectory_time_step_counter=self.current_trajectory_time_step_counter,
                agent_gas_price_gwei=agent_gas_price_bid
            )
        else:
            current_fee = 0.0
            trade_occured = False

        idx_current_asset_invested = self.generate_allocation_decoding(self.current_state_portfolio_allocation)

        # End of period /here we evaluate the position that was held during the time step
        str_trading_pair = self.get_str_trading_pair(idx_current_asset_invested)

        econ_reward = float(
            self.current_trajectory_df.iloc[
                self.current_trajectory_time_step_counter + 1
            ][f"{str_trading_pair}_close_to_close"]
        )

        current_econ_reward = float(
            self.current_trajectory_df.iloc[
                self.current_trajectory_time_step_counter + 1
            ][f"{str_trading_pair}_close_to_close"]
        )

        # 1st fees are applied, 2nd the econ rewards are applied which is approximately to (1-fee)*(1+econ_reward)-1
        if current_fee >= 1.0:
            warnings.warn(f'Unexpected high fee of {current_fee*100:.2f}% relative to the currently available funds'
                          f' of {self.current_state_funds:.4f} which resulted in bankruptcy this episode.'
                          f' Consider reducing the "fee_difficulty_scaling_factor".')
            self.current_state_funds = 0.0  # bankrupt
            ln_current_fee = -10  # math.log(1-0.9999)
        else:
            ln_current_fee = math.log(1 - current_fee)

        if current_econ_reward <= -1.0:
            warnings.warn(f'Unexpected high negative return of  {current_econ_reward*100:.2f}%'
                          f' which lead to bankruptcy for this episode".')
            self.current_state_funds = 0.0  # bankrupt
            ln_current_econ_reward = -10
        else:
            ln_current_econ_reward = math.log(1 + current_econ_reward)

        ln_reward = ln_current_econ_reward + ln_current_fee

        reward = math.exp(ln_reward) - 1

        """
        #ln_current_econ_reward
        #reward = (
        #    current_econ_reward
        #    - current_fee
        #)  # generate r_{t+1}

        # FIXME account for case that current_fee is > 100%
        try:
            reward_ln = math.log(1.0 + reward)
        except:
            raise ValueError(f'Unexpected high negative reward of {reward*100:.2f}%. '
                             f'Check the current fees which account for {current_fee*100:.2f}% of the currently available funds'
                             f' of {self.current_state_funds:.4f}. Consider reducing the "fee_difficulty_scaling_factor".')
        """ or None
        self.current_state_funds = self.current_state_funds * (1.0 + reward)

        observation_ = self.generate_observation(
            self.current_state_funds,
            self.current_state_portfolio_allocation,
            current_trajectory_time_step=self.current_trajectory_time_step_counter + 1,
        )  # to generate s_{t+1}

        self.current_trajectory_time_step_counter += 1

        if (
            self.current_trajectory_time_step_counter + 1
            < self.amount_time_steps_to_include
        ):  # +1 is necessary due to the 0-idx
            done = False
        else:
            done = True

        info_dict = {
            "fee": current_fee,
            "econ_return": current_econ_reward,
            "trade_occured": trade_occured,
        }

        if self.include_gas_bid_in_action:
            info_dict["agent_gas_price_bid"] = agent_gas_price_bid

        return observation_, ln_reward, done, info_dict

    def generate_observation(
        self,
        current_state_funds: float,
        current_state_portfolio_allocation: np.ndarray,
        current_trajectory_time_step: int,
    ) -> np.ndarray:

        return np.concatenate(
            (
                [current_state_funds],
                self.current_trajectory_fee_df.to_numpy()[
                    current_trajectory_time_step, :
                ],
                current_state_portfolio_allocation,
                self.current_trajectory_df.to_numpy()[current_trajectory_time_step, :],
            )
        )

    def reset(self) -> np.ndarray:
        self.current_initial_time_step = self.sample_allowed_initial_time_index()

        self.current_trajectory_time_step_counter = 0
        self.current_trajectory_df = self.build_current_trajectory_data(
            initial_time_index=self.current_initial_time_step
        )
        self.current_trajectory_fee_df = self.build_current_trajectory_fee_data(
            initial_time_index=self.current_initial_time_step
        )
        self.current_state_funds = self.starting_funds
        self.current_state_portfolio_allocation = self.generate_allocation_encoding(
            0
        )  # we start with ETH/ETH position

        return self.generate_observation(
            self.current_state_funds,
            self.current_state_portfolio_allocation,
            current_trajectory_time_step=self.current_trajectory_time_step_counter,
        )
