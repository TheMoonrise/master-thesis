from setuptools import setup

setup(
    name="crypto_markets_gym",  # be aware of _ and -, the root folder is - while the subfolder is _
    version="1.0.0",
    install_requires=["gym", "numpy", "pandas", "pyarrow"],
)
