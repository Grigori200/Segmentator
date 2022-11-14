import os

from segmentation import sole_limitation
import pandas as pd


def main():
    fifth = sole_limitation(os.path.join('data', 'odc2.wav'), 'outputs', 30.)
    print(pd.read_csv(os.path.join(fifth, 'timestamps.csv'), index_col=0))


if __name__ == '__main__':
    main()
