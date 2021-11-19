import pandas as pd
import argparse
from dataclasses import dataclass
import json
from collections import defaultdict

@dataclass
class Args:
    input: str
    output: str

def get_args():
    parser = argparse.ArgumentParser(description='Make csv dataset')
    parser.add_argument('-i', '--input', type=str, required=True, help='jsonl input file')
    parser.add_argument('-o', '--output', type=str, required=True, help='output file')

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
                data[k].append(v)

    df = pd.DataFrame.from_dict(data)

    # write the output
    df.to_csv(args.output, index=False)


if __name__ == '__main__':
    main(get_args())