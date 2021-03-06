import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import os
from glob import glob
from pathlib import Path


# describe table + nan count, nan pct and unique count
def summary(df):
    summary_df = df.describe()
    if 'top' in summary_df.index.values:
        summary_df.drop(['top', 'freq', 'unique'], axis=0, inplace=True)

    nan_count = df.isnull().sum().to_frame().transpose().rename(index={0: 'NAN count'})
    nan_pct = (df.isnull().sum() / len(df.index) * 100).to_frame().transpose().rename(index={0: 'NAN percent'})
    nunique = df.nunique().to_frame().transpose().rename(index={0: 'unique'})
    rep = ((len(df.index) - df.isnull().sum()) / df.nunique()).to_frame().transpose().rename(index={0: 'repetition'})

    summary_df = summary_df.append([nunique, nan_count, nan_pct, rep], sort=False)

    return summary_df


def summary_database(engine, tables):
    if not os.path.exists('tmp'):
        os.makedirs('tmp')

    current_files = glob.glob("tmp/*.csv")
    seen_tables = [Path(x).stem for x in current_files]

    empty_table_path = "tmp/empty_tables.txt"
    if os.path.exists(empty_table_path):
        with open(empty_table_path, 'r') as f:
            content = f.read()
            seen_tables.extend(content.split())

    with tqdm(tables) as pbar:
        for t in tables:
            pbar.set_postfix(DB=t, refresh=True)
            table_file = "tmp/" + t + ".csv"
            if t in seen_tables:
                time.sleep(0.1)
                pbar.update(1)
                continue

            df = pd.read_sql('SELECT * FROM ' + t, engine)
            if len(df.index) == 0:
                append_write = 'a' if os.path.exists(empty_table_path) else 'w'
                with open(empty_table_path, append_write) as f:
                    f.write(t + "\n")
                pbar.update(1)
                continue
            sm = summary(df).T
            sm['column'] = sm.index
            sm['table'] = t
            cols = sm.columns.values.tolist()
            sm = sm[['table', 'column'] + cols[:-2]]
            sm.to_csv(table_file, index=False)
            pbar.update(1)

    empty_tables = []
    with open(empty_table_path, 'r') as f:
        content = f.read()
        empty_tables = content.split()

    res = pd.DataFrame()
    with tqdm(tables) as pbar:
        for t in tables:
            table_file = "tmp/" + t + ".csv"
            if t in empty_tables:
                time.sleep(0.1)
                pbar.update(1)
                continue

            df = pd.read_csv(table_file, index_col=None, header=0)
            res = res.append(df, ignore_index=True)
            pbar.update(1)

    return res, empty_tables


def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2
    print("Memory usage of properties dataframe is :" ,start_mem_usg ," MB")
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()

            # If exists NAN value, skip it
            if not np.isfinite(df[col]).all() or np.isnan(df[col].any()):
                continue

            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)

                        # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2
    print("Memory usage is: " ,mem_usg ," MB")
    print("This is ",100 *mem_usg /start_mem_usg ,"% of the initial size")
    return df
