import math

import numpy as np

import json
import gym
from typing import Tuple, Dict, List
import importlib.resources
import json
import pandas as pd
import random


class FeeModel:
    FULL_FEE = "full_fee"
    NO_FEE = "no_fee"


class CryptoMarketsEnv(gym.Env):

    LIST_STATE_RELEVANT_ASSET_COLUMNS = [
        "vol_token0",
        "vol_token1",
        "tx",
        "max_to_close",
        "min_to_close",
        "close_to_close",
    ]

    LIST_STATE_RELEVANT_FEE_COLUMNS = ["avg_gas_price_per_block_gwei"]

    def __init__(
        self,
        config,
        **kwargs,
    ):

        # NOTE input parameters replaced by config
        # parameters are now fetched from the config below
        amount_time_steps_to_include = (12 * 24)
        starting_funds = 1.0
        fee_model = FeeModel.FULL_FEE

        self.meta_actions = False

        if 'amount_time_steps_to_include' in config: amount_time_steps_to_include = config['amount_time_steps_to_include']
        if 'starting_funds' in config: starting_funds = config['starting_funds']
        if 'fee_model' in config: fee_model = config['fee_model']
        if 'meta_actions' in config: self.meta_actions = config['meta_actions']

        print(f"STARTING FUNDS {starting_funds}")

        self.amount_pairs_to_include = 10

        self.total_crypto_data = self.initializing_crypto_data()
        self.total_crypto_fee_data = self.initializing_crypto_fee_data()
        self.list_relevant_pairs = self.initialize_list_relevant_pairs()
        self.total_amount_observations_per_asset = self.total_crypto_data["index"].max()

        self.fee_model = fee_model

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

        # FIXME only valid for the prototype environment
        self.action_space = gym.spaces.Discrete(self.amount_pairs_to_include)

        # NOTE modify the action space if meta actions are enabled
        if self.meta_actions: self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.amount_pairs_to_include,))

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

    def sample_allowed_initial_time_index(self) -> int:
        return random.randint(
            0,
            (
                self.total_amount_observations_per_asset
                - self.amount_time_steps_to_include
                - 1
            ),
        )  # -1 due to 0 indexing

    def initializing_crypto_fee_data(self) -> pd.DataFrame:

        # https://stackoverflow.com/questions/779495/access-data-in-package-subdirectory
        with importlib.resources.open_binary(
            "crypto_markets_gym.envs.data", f"crypto_fee_data_total.ftr"
        ) as file:
            columns_to_read = [
                "index",
                "associated_time_slot",
                "avg_gas_price_per_block_gwei",
            ]

            crypto_fee_df = pd.read_feather(file, columns=columns_to_read)

        return crypto_fee_df

    def initializing_crypto_data(self) -> pd.DataFrame:

        # https://stackoverflow.com/questions/779495/access-data-in-package-subdirectory
        with importlib.resources.open_binary(
            "crypto_markets_gym.envs.data", f"crypto_pool_data_total.ftr"
        ) as file:
            columns_to_read = [
                "index",
                "associated_time_slot",
                "token0",
                "vol_token0",
                "token1",
                "vol_token1",
                "tx",
                "trading_pair",
                "max_to_close",
                "min_to_close",
                "close_to_close",
            ]

            crypto_df = pd.read_feather(file, columns=columns_to_read)

        return crypto_df

    def initialize_list_relevant_pairs(self) -> List[Dict]:

        pair_file = "uni_v3_pairs.json"

        with importlib.resources.open_text(
            "crypto_markets_gym.envs.data", f"{pair_file}"
        ) as file:
            list_pairs = json.load(file)

        tmp_list_unique = self.total_crypto_data["trading_pair"].unique()

        list_relevant_pairs = [
            val
            for ind, val in enumerate(list_pairs)
            if (
                f'{val.get("token0").get("symbol")}_{val.get("token1").get("symbol")}'
                in tmp_list_unique
            )
            & (ind < self.amount_pairs_to_include)
        ]
        return list_relevant_pairs

    def calculate_current_fee(
        self,
        current_state_funds,
        previous_state_portfolio_allocation,
        current_state_portfolio_allocation,
        current_trajectory_time_step_counter,
    ) -> float:
        if self.fee_model == FeeModel.NO_FEE:
            return 0.0
        elif self.fee_model == FeeModel.FULL_FEE:
            # we assume a gas cost of 160_000
            # (160_000/1_000_000_000)*GWEI_PRICE #assuming that we have 160_000 gas cost for an Asset1 -> ETH -> Asset2 trip
            trading_delta_vector = (
                current_state_portfolio_allocation - previous_state_portfolio_allocation
            )
            previous_idx_asset = np.argmax(previous_state_portfolio_allocation)
            current_idx_asset = np.argmax(current_state_portfolio_allocation)
            if np.sum(np.abs(trading_delta_vector)) > 0:
                previous_trading_pair_trading_fee = (
                    float(self.list_relevant_pairs[previous_idx_asset].get("feeTier"))
                    / 1000000.0
                )  # measured in 1/millionth
                current_trading_pair_trading_fee = (
                    float(self.list_relevant_pairs[current_idx_asset].get("feeTier"))
                    / 1000000.0
                )  # measured in 1/millionth
                # We charge as gas/gwei the price in the previous observed 5min window
                current_gas_price_ETH = float(
                    self.current_trajectory_fee_df.to_numpy()[
                        current_trajectory_time_step_counter, :
                    ]
                    / 1_000_000_000
                )  # measured in gwei
                fixed_network_fee_swap = (
                    160_000 * current_gas_price_ETH
                )  # assumption of 160_000 gas required per swap
                total_fee_relative = fixed_network_fee_swap / current_state_funds + (
                    previous_trading_pair_trading_fee + current_trading_pair_trading_fee
                )
                return total_fee_relative
            else:
                return 0.0  # no trade occured

    def build_current_trajectory_data(self, initial_time_index: int) -> pd.DataFrame:
        tmp_list_dfs = []

        for key_pair in [
            f'{entry.get("token0").get("symbol")}_{entry.get("token1").get("symbol")}'
            for entry in self.list_relevant_pairs
        ]:

            tmp_df = self.total_crypto_data[
                self.total_crypto_data["trading_pair"] == key_pair
            ]
            # slicing the required time index
            tmp_df = tmp_df[
                (tmp_df["index"] >= initial_time_index)
                & (
                    tmp_df["index"]
                    < (initial_time_index + self.amount_time_steps_to_include)
                )
            ]

            tmp_df = tmp_df.set_index("index")
            tmp_df = tmp_df[CryptoMarketsEnv.LIST_STATE_RELEVANT_ASSET_COLUMNS]
            tmp_df.columns = [f"{key_pair}_{tmp_entry}" for tmp_entry in tmp_df.columns]
            tmp_list_dfs.append(tmp_df)

        merged_df = pd.concat(tmp_list_dfs, axis=1)
        return merged_df

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

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:

        # NOTE if meta actions are enabled the action is sampled here
        if self.meta_actions: action = np.array(np.random.choice(self.amount_pairs_to_include, p=action))

        # Beginning of time step / here we do potential trading
        idx_current_asset_invested = (
            action.item()
        )  # index of the current asset invested
        previous_state_portfolio_allocation = self.current_state_portfolio_allocation
        self.current_state_portfolio_allocation = self.generate_allocation_encoding(
            idx_current_asset_invested
        )
        current_fee = self.calculate_current_fee(
            current_state_funds=self.current_state_funds,
            previous_state_portfolio_allocation=previous_state_portfolio_allocation,
            current_state_portfolio_allocation=self.current_state_portfolio_allocation,
            current_trajectory_time_step_counter=self.current_trajectory_time_step_counter,
        )

        # End of period /here we evaluate the position that was held during the time step
        str_trading_pair = (
            f'{self.list_relevant_pairs[idx_current_asset_invested].get("token0").get("symbol")}_'
            f'{self.list_relevant_pairs[idx_current_asset_invested].get("token1").get("symbol")}'
        )

        reward = (
            float(
                self.current_trajectory_df.iloc[
                    self.current_trajectory_time_step_counter + 1
                ][f"{str_trading_pair}_close_to_close"]
            )
            - current_fee
        )  # generate r_{t+1}
        self.current_state_funds = self.current_state_funds * (1.0 + reward)
        reward_ln = math.log(1.0 + reward)

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

        return observation_, reward_ln, done, {}

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
