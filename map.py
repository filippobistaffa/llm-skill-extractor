import argparse as ap
import pandas as pd
import numpy as np
import os


if __name__ == "__main__":

    # parse command-line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('--embeddings', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'skills-embeddings-3-small.tar.gz'))
    parser.add_argument('--skills', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'clean-0.csv'))
    args, additional = parser.parse_known_args()

    embeddings = pd.read_pickle(args.embeddings)
    print(embeddings)
