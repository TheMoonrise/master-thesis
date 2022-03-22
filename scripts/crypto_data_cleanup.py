import os
import os.path as path
import pandas as pd
import argparse


def extracted_data_from_file(file, start, end, interval):
    """
    Extracts the data from a given .csv file.
    :param file: The path to the file to be read.
    :param start: The start timestamp for data collection.
    :param end: The end timestamp for data collection.
    :param interval: The interval at which data is aggregated.
    :return: A list of market data in the given interval.
    """
    print('Processing ' + path.basename(file))
    df = pd.read_csv(file)

    last_row = None
    timestep = start
    data = []

    for i, row in df.iterrows():
        while row['time'] > timestep:
            data.append(last_row['close'])
            timestep += interval

        last_row = row
        print(f'({i:0>{len(str(df.shape[0]))}}/{df.shape[0]})', end='\r')
        if (timestep > end): break

    return data


def symbol_from_path(file):
    """
    Retrieves the coin symbol from the .csv file name.
    :param file: The path to the file.
    :return: The coin symbol as string.
    """
    symbol = path.basename(file)
    symbol = symbol.replace('usd', '').replace('-', '').replace('.csv', '')
    return symbol.upper()


if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    # require arguments for the input directory and the output file
    parse.add_argument('path', help='The path to the folder containing the csv files', type=str)
    parse.add_argument('out', help='The path to the output file', type=str)

    # require arguments for the start and end timestamp between which data will be collected
    parse.add_argument('start', help='The UNIX start timestamp from which data will be collected', type=int)
    parse.add_argument('end', help='The UNIX end timestamp until which data will be collected', type=int)
    parse.add_argument('--interval', help='The data interval in ms', type=int, default=300_000)

    args = parse.parse_args()

    # validate that the given path is valid
    if (not path.isdir(args.path) or not path.exists(args.path)):
        raise Exception('The given source path is invalid')

    if (not path.exists(path.dirname(args.out))):
        raise Exception('The given output file path is invalid')

    # find all .csv files in the given directory
    files = [path.join(args.path, f) for f in os.listdir(args.path) if f.endswith('.csv')]

    # define an empty data frame that will be filled with market data
    df = pd.DataFrame({'timestamp': [x for x in range(args.start, args.end + 1, args.interval)]})

    # iterate all files and append them to the data frame
    for f in files:
        symbol = symbol_from_path(f)
        data = extracted_data_from_file(f, args.start, args.end, args.interval)
        df[symbol] = data

    # write the completed dataframe to .csv
    df.to_csv(args.out, index=False)
