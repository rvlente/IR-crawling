import pandas as pd
import argparse
from dataclasses import dataclass
import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm

@dataclass
class Args:
    input: str
    output: str
    debug_mode: bool

def get_args():
    parser = argparse.ArgumentParser(description='Make csv dataset')
    parser.add_argument('-i', '--input', type=str, required=True, help='jsonl input file')
    parser.add_argument('-o', '--output', type=str, required=True, help='output file')
    parser.add_argument('--debug-mode', action='store_true', help='debug mode')

    args = Args(**vars(parser.parse_args()))
    return args

def main(args):
    # read the input
    data = defaultdict(list)
    with open(args.input, 'r') as f:
        lines = f.readlines()
        for line in lines:
            try:
                entry = json.loads(line)
            except json.decoder.JSONDecodeError:
                entry = {}
                pass
            for k, v in entry.items():
                # if isinstance(v, str):
                #     # remove invalid utf-8 characters
                #     v = v.encode('utf-8', errors='ignore').decode('utf-8')
                # if isinstance(v, bool):
                #     v = 1 if v else 0
                data[k].append(v)

    

    df = pd.DataFrame.from_dict(data)

    # drop duplicates of the 'url' field
    df = df.drop_duplicates(subset='url')

    # write the output
    # df.to_csv(args.output, index=False)
    df.to_parquet(args.output, index=False)

    if args.debug_mode:
        
        # saved_df = pd.read_csv(args.output)
        saved_df = pd.read_parquet(args.output)

        # replace all NaNs with empty strings
        df = df.fillna('')
        saved_df = saved_df.fillna('')

        # assert df.equals(saved_df)

        # zip over rows
        for row_df, row_saved_df in tqdm(zip(df.iterrows(), saved_df.iterrows()), total=len(df)):
            # zip over columns
            for col_df, col_saved_df in zip(row_df[1].iteritems(), row_saved_df[1].iteritems()):
                assert col_df[1] == col_saved_df[1], f'{col_df} != {col_saved_df}'
            # assert (row_df == row_saved_df).all(), f'{row_df} != {row_saved_df}'



if __name__ == '__main__':
    main(get_args())