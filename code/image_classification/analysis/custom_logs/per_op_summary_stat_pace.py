# %%
import os,argparse,natsort
import pandas as pd

# display more columns
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# %%

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str,\
                     default='/home/mayurpl/sem_2/special_problems/ml_profiling/pace_logs/results',\
                        help="path to the directory containing the log files")
# for percentile
parser.add_argument("--percentile", type=float, default=0.9,\
                        help="percentile to be calculated (between 0 and 1)")
args = parser.parse_args()


# percentile function
def percentile_func(x):
    return x.quantile(args.percentile)

def generate_summary_stats_per_op(data_dir):

    root_to_files = {}
    for root, dirs, files in os.walk(data_dir):
        root_to_files[root] = files
    roots = sorted(root_to_files, key=lambda x: natsort.natsort_key(x.lower()))

    # recursively find all the log files
    for root in roots:
        if 'e2e' in root:
            continue
        files = root_to_files[root]
        print(root)
        combine_df = pd.DataFrame()
        for file in files:
            if 'worker_pid' in file:
                fp = os.path.join(root, file)
                #  read the log file as csv with no header
                df = pd.read_csv(fp, header=None)
                df.columns = ["name", "start", "dur"]

                # drop columns with name "start"
                df = df.drop(columns=["start"])

                # divide "dur" column by 1000,000 to get time in ms
                df["dur"] = df["dur"] / 1000000
                # discard the rows with name starting with SBatch
                df = df[~df["name"].str.startswith("SBatch")]

                # concatenate all the dataframes
                combine_df = pd.concat([combine_df, df])

        # check if the dataframe is empty
        if combine_df.empty:
            continue

        # print mean, std and P99 for each operation
        print(f"Percentile: {args.percentile}")
        stats_df = combine_df.groupby("name").agg(["sum","mean", "median", percentile_func, "std"])
        print(stats_df.head(len(stats_df)))
        print("\n\n")

        # %%
        print("% of each operation with duration < 10 ms")
        print(pd.DataFrame(combine_df[combine_df["dur"] < 10].groupby("name").count() * 100 / combine_df.groupby("name").count()).fillna(0))
        print("\n\n")

        # %%
        print("% of each operation with duration < 100 us")
        print(pd.DataFrame(combine_df[combine_df["dur"] < 0.1].groupby("name").count() * 100 / combine_df.groupby("name").count()).fillna(0))
        print("--------------------------------------------------\n\n")
# %%
print("All numbers are in ms")
generate_summary_stats_per_op(args.data_dir)