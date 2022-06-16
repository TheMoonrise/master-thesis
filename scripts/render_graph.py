"""
Draws graphs from given data.
"""
import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

parse = argparse.ArgumentParser()

parse.add_argument('source', type=str)
parse.add_argument('property', type=str)

args = parse.parse_args()

# read all data files from the source folder
graphs = {}

for file in os.listdir(args.source):
    if not file.endswith('.json'): continue

    with open(os.path.join(args.source, file)) as stream:
        data = json.loads(stream.read())

    if not args.property in data: continue
    graphs[file.replace('.json', '')] = data[args.property]

print(graphs)
fig, ax = plt.subplots()
for v in graphs.values(): ax.plot(v)
plt.show()
