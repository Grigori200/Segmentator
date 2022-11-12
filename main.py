import os

from segmentation import full_segmentation
import pandas as pd


def main():
    # first = full_segmentation(os.path.join('data', '2007100212545620.wav'), 'outputs', 30., 1.)
    # second = full_segmentation(os.path.join('data', 'a1.wav'), 'outputs', 30., 1.)
    # third = full_segmentation(os.path.join('data', 'a2.wav'), 'outputs', 30., 1.)
    # fourth = full_segmentation(os.path.join('data', 'Jeziorany_odc1.wav'), 'outputs', 30., 1.)
    fifth = full_segmentation(os.path.join('data', 'odc2.wav'), 'outputs', 30., 1.5)
    # print(pd.read_csv(os.path.join(first, 'timestamps.csv'), index_col=0))
    # print(pd.read_csv(os.path.join(second, 'timestamps.csv'), index_col=0))
    # print(pd.read_csv(os.path.join(third, 'timestamps.csv'), index_col=0))
    # print(pd.read_csv(os.path.join(fourth, 'timestamps.csv'), index_col=0))
    print(pd.read_csv(os.path.join(fifth, 'timestamps.csv'), index_col=0))


if __name__ == '__main__':
    main()
