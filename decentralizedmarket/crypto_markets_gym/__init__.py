from gym.envs.registration import register

register(
    id="crypto-markets-env-v0",
    entry_point="crypto_markets_gym.envs:CryptoMarketsEnv",
)
