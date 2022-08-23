import pandas as pd
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--symbol', type=str, default='symbol')
parser.add_argument('--csv_file', type=str, required=True)
parser.add_argument('--csv_dir', type=str, required=True)
parser.add_argument('--limit_stocks', type=int, required=False, default=None)
args = parser.parse_args()

symbol = args.symbol
csv_file = args.csv_file
csv_dir = args.csv_dir
df = pd.read_csv(csv_file)
codes = df[symbol].unique()
limit = args.limit_stocks if args.limit_stocks is not None else len(codes)
for c in tqdm(codes[:limit], desc='Creating stock files'):
    df_code = df[df[symbol] == c].reset_index(drop=True)
    df.to_csv(os.path.join(csv_dir, c + '.csv'), encoding='utf-8', index=False)
