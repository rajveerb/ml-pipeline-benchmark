# %%
# open a file as csv without header

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,natsort
import argparse

plt.rc('axes', labelsize=95)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=70)    # fontsize of the tick labels
plt.rc('ytick', labelsize=70)    # fontsize of the tick labels
plt.rc('legend', fontsize=60)    # legend fontsize

figsize = (50,25)

# take below arguments using argparse
# add argument to pass pytorch_profiler_data_file
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str,
                    default='/home/mayurpl/sem_2/special_problems/ml_profiling/pace_logs/results', help='Root directory with custom_log for different configs')

parser.add_argument('--sort_criteria', type=str,
                    default='batch_id', help='sort by `batch_id` or `duration`')

parser.add_argument('--fig_dir', type=str,
                    default='./figs_pace',
                    help='Path to store the figures')
                    

args = parser.parse_args()

# %%
# sort_by = 'batch_id' or 'duration'
def plotter_preprocessing_time(target_dir,sort_by='batch_id',fig_size=(50,25),remove_outliers=True,fig_prefix='',fig_dir=''):
    
    root_to_files = {}
    for root, dirs, files in os.walk(target_dir):
        root_to_files[root] = files
    roots = sorted(root_to_files, key=lambda x: natsort.natsort_key(x.lower()))

    prev_batch = None
    for root in roots:
        if 'e2e' in root:
            continue
        print(root)
        files = root_to_files[root]
        plot_df = pd.DataFrame()
        
        for file in files:
            if "worker_pid" not in file:
                continue

            df = pd.read_csv(os.path.join(root, file)
                            , header=None)

            # add header
            df.columns = ['name','start_ts','duration']

            # names that start with 'SBatchPreprocessed'
            df = df[df['name'].str.startswith('SBatchPreprocessed')]
            # map 'SBatchPreprocessed_' such that 'SBatchPreprocessed_idx' becomes 'idx' where idx is an integer
            df['batch_id'] = df['name'].map(lambda x: int(x.replace('SBatchPreprocessed_','')))


            # divide by 1000000 to convert from nanoseconds to milliseconds
            df['duration'] = df['duration']/1000000

            # concatentate all dataframes
            plot_df = pd.concat([plot_df, df])
        
        if plot_df.empty:
            continue
        def remove_wild_outliers(plot_df):
            # mean and std before removing outliers
            mean = np.mean(plot_df["duration"])
            std = np.std(plot_df["duration"])
            total_preprocessing_time = np.sum(plot_df["duration"])
            print (f'Before removing outliers:')
            print (f'sum = {total_preprocessing_time:.2f} ms, avg = {mean:.2f} ms, std = {std:.2f} ({100*std/mean:.2f}% of avg) ms , min = {plot_df["duration"].min():.2f} ms, max = {plot_df["duration"].max():.2f} ms')
            # print median, 25th and 75th percentile
            print (f'median = {np.median(plot_df["duration"]):.2f} ms, 25th percentile = {np.percentile(plot_df["duration"], 25):.2f} ms, 75th percentile = {np.percentile(plot_df["duration"], 75):.2f} ms')
            print(f"75th - 25th = {np.percentile(plot_df['duration'], 75) - np.percentile(plot_df['duration'], 25):.2f} ms")
            
            q1 = plot_df['duration'].quantile(0.25)
            q3 = plot_df['duration'].quantile(0.75)
            iqr = (q3 - q1) * 2

            # remove outliers less than q1 - iqr only (these are numbers from last batch which has
            #  elements less than batch size because elements in a dataset may not be a multiple of batch size)
            print("remove outliers less than q1 - iqr, where iqr = (q3 - q1) * 2")
            plot_df = plot_df[~(plot_df['duration'] < (q1 - iqr))]
            # mean and std after removing outliers
            print (f'After removing outliers:')
            return plot_df
        batch = root.split('/')[-1].split('b')[1].split('_')[0] # <long_dir_path> -> 128_gpu4 -> 128

        label = root.split('/')[-1] # retrieves b128_gpu4 kind of label
        print(f'{label}:')
        # remove outliers
        if remove_outliers:
            plot_df = remove_wild_outliers(plot_df)

        if prev_batch is None:
            prev_batch = batch 
        elif batch != prev_batch:

            # plt.figure(figsize=fig_size)
            plt.gcf().set_size_inches(fig_size[0], fig_size[1])
            # label x axis
            plt.xlabel(f'Batch ids preprocessed (sorted by {sort_by}) (batch size = {prev_batch})', labelpad=40)
            # label y axis
            plt.ylabel('Preprocessing time in ms', labelpad=40)
            plt.tick_params(axis='y', which='major', pad=40) 
            plt.tick_params(axis='x', which='major', pad=40) 
            # add legend to bottom right and increase size of legend markers
            plt.legend(loc='lower right',markerscale=4)
            plt.tight_layout()
            fig_path = os.path.join(fig_dir, f'{fig_prefix}{prev_batch}_batch_preprocessing_time_{sort_by}.png') 
            plt.savefig(fig_path)
            plt.clf()
            prev_batch = batch            
        
        mean = np.mean(plot_df["duration"])
        std = np.std(plot_df["duration"])
        total_preprocessing_time = np.sum(plot_df["duration"])
        print (f'sum = {total_preprocessing_time:.2f} ms, avg = {mean:.2f} ms, std = {std:.2f} ({100*std/mean:.2f}% of avg) ms, min = {plot_df["duration"].min():.2f} ms, max = {plot_df["duration"].max():.2f} ms')
        # print median, 25th and 75th percentile
        print (f'median = {np.median(plot_df["duration"]):.2f} ms, 25th percentile = {np.percentile(plot_df["duration"], 25):.2f} ms, 75th percentile = {np.percentile(plot_df["duration"], 75):.2f} ms')
        print(f"75th - 25th = {np.percentile(plot_df['duration'], 75) - np.percentile(plot_df['duration'], 25):.2f} ms")

        plot_df = plot_df.sort_values(by=[sort_by])
        if sort_by == 'batch_id':
            plt.plot(plot_df['batch_id'], plot_df['duration'], label=label, marker='s')
        else:
            # reset index
            plot_df = plot_df.reset_index(drop=True)
            plt.scatter(plot_df.index, plot_df['duration'], label=label, marker='s')
            # remove xticks
            plt.xticks([])
        # plot on log scale
        plt.yscale('log')

    # plt.figure(figsize=fig_size)
    plt.gcf().set_size_inches(fig_size[0], fig_size[1])
    # label x axis
    plt.xlabel(f'Batch ids preprocessed (sorted by {sort_by})', labelpad=40)
    # label y axis
    plt.ylabel('Preprocessing time in ms', labelpad=40)
    plt.tick_params(axis='y', which='major', pad=40) 
    plt.tick_params(axis='x', which='major', pad=40) 
    # add legend to bottom right and increase size of legend markers
    plt.legend(loc='lower right',markerscale=4)
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, f'{fig_prefix}{prev_batch}_batch_preprocessing_time_{sort_by}.png') 
    plt.savefig(fig_path)
    plt.clf()

# %%
# sort_by = 'name' or 'duration'
def plotter_preprocessing_wait_time(target_dir,sort_by='batch_id',fig_size=(50,25),fig_prefix='',fig_dir=''):
    plt.figure(figsize=fig_size)
    root_to_files = {}
    for root, dirs, files in os.walk(target_dir):
        root_to_files[root] = files
    roots = sorted(root_to_files, key=lambda x: natsort.natsort_key(x.lower()))
    
    prev_batch = None
    
    for root in roots:
        if 'e2e' in root:
            continue
        print (root)
        files = root_to_files[root]
        plot_df = pd.DataFrame()
        for file in files:
            if "main_pid" not in file:
                continue

            df = pd.read_csv(os.path.join(root, file)
                            , header=None)

            # add header
            df.columns = ['name','start_ts','duration']

            # names that start with 'SBatchWait'
            df = df[df['name'].str.startswith('SBatchWait')]

            # map 'SBatchWait' such that 'SBatchWait_idx' becomes 'idx' where idx is an integer
            df['batch_id'] = df['name'].map(lambda x: int(x.replace('SBatchWait_','')))

            # divide by 1000000 to convert from nanoseconds to milliseconds
            df['duration'] = df['duration']/1000000

            # concatentate all dataframes
            plot_df = pd.concat([plot_df, df])
        
        if plot_df.empty:
            continue

        label = root.split('/')[-1] # retrieves b128_gpu4 kind of label
        batch = root.split('/')[-1].split('b')[1].split('_')[0] # <long_dir_path> -> 128_gpu4 -> 128

        print(f'{label}:')

        # print % of batches for which main process had to wait > 1 us and less than 1 ms pretty print
        equal_to_1us = len(plot_df[plot_df["duration"] == 0.001])
        less_than_1ms = len(plot_df[plot_df["duration"] < 1])
        greater_than_1ms = len(plot_df[plot_df["duration"] > 1])
        print (f'{100*(less_than_1ms-equal_to_1us)/len(plot_df):.2f}% of batches had 1 us < wait time < 1 ms')
        print (f'{100*greater_than_1ms/len(plot_df):.2f}% of batches had wait time > 1 ms')

        # print % of batches for which main process had to wait == 1 us pretty print
        print (f'{100*equal_to_1us/len(plot_df):.2f}% of batches had wait time == 1 us')

        # if wait_time is greater than 1 ms, then it is a good data point
        # plot_df = plot_df[plot_df['duration'] > 1]
        if prev_batch is None:
            prev_batch = batch 
        elif batch != prev_batch:

            # plt.figure(figsize=fig_size)
            plt.gcf().set_size_inches(fig_size[0], fig_size[1])
            # label x axis
            plt.xlabel(f'Batch ids preprocessed (sorted by {sort_by}) (batch size = {prev_batch})', labelpad=40)
            # label y axis
            plt.ylabel('Wait time in ms', labelpad=40)
            plt.tick_params(axis='y', which='major', pad=40) 
            plt.tick_params(axis='x', which='major', pad=40) 
            # add legend to bottom right and increase size of legend markers
            # plt.legend(loc='lower right',markerscale=4)
            # put legend outside the plot
            # plt.legend(bbox_to_anchor=(1.05, 1), markerscale=4)
            plt.tight_layout()
            fig_path = os.path.join(fig_dir,f'{fig_prefix}{prev_batch}_batch_preprocessing_wait_time.png')
            plt.savefig(fig_path)
            plt.clf()
            prev_batch = batch 

        mean = np.mean(plot_df["duration"])
        std = np.std(plot_df["duration"])
        minimum = plot_df["duration"].min()
        maximum = plot_df["duration"].max()
        total_wait_time = np.sum(plot_df["duration"])
        print (f'sum = {total_wait_time:.2f} ms, avg = {mean:.2f} ms, std = {std:.2f} ({100*std/mean:.2f}% of avg) ms, min = {minimum:.2f} ms, max = {maximum:.2f} ms')
        # print % of batches for which main process had to wait > 500 ms pretty print
        print (f'{100*len(plot_df[plot_df["duration"] > 500])/len(plot_df):.2f}% of batches had wait time > 500 ms')
        
        # sort by sort_by
        plot_df = plot_df.sort_values(by=[sort_by])
        plt.scatter(plot_df['batch_id'], plot_df['duration'], label=label,s=300)
        # plot on log scale
        plt.yscale('log')

    plt.gcf().set_size_inches(fig_size[0], fig_size[1])
    # label x axis
    plt.xlabel(f'Wait time for batch ids preprocessed (sorted by {sort_by})', labelpad=40)
    # label y axis
    plt.ylabel('Wait time in ms', labelpad=40)
    plt.tick_params(axis='y', which='major', pad=40) 
    plt.tick_params(axis='x', which='major', pad=40) 
    # add legend to bottom right and increase size of legend markers
    # plt.legend(loc='lower right',markerscale=4)
    # plt.legend(bbox_to_anchor=(1.05, 1), markerscale=4)
    plt.tight_layout()
    fig_path = os.path.join(fig_dir,f'{fig_prefix}{prev_batch}_batch_preprocessing_wait_time.png')
    plt.savefig(fig_path)
    plt.clf()

# %%
# sort_by = 'batch_id' or 'duration'
def plotter_diff_consumed_preprocess_end_per_batch_time(target_dir,sort_by='batch_id',fig_size=(50,25),remove_outliers=True,fig_prefix='',fig_dir=''):
    
    if sort_by == 'duration':
        sort_by = 'wait_time'

    root_to_files = {}
    for root, dirs, files in os.walk(target_dir):
        root_to_files[root] = files
    roots = sorted(root_to_files, key=lambda x: natsort.natsort_key(x.lower()))

    prev_batch = None
    for root in roots:
        if 'e2e' in root:
            continue
        print(root)
        files = root_to_files[root]
        worker_df = pd.DataFrame()
        main_df = pd.DataFrame()
        final_df = pd.DataFrame()
        if 'b512' not in root:
            continue
        for file in files:
            if "main_pid" in file:
                main_df = pd.read_csv(os.path.join(root, file)
                            , header=None)

                # add header
                main_df.columns = ['name','consume_ts','duration']

                # names that start with 'SBatchConsumed'
                main_df = main_df[main_df['name'].str.startswith('SBatchConsumed')]
                # map 'SBatchConsumed_' such that 'SBatchConsumed_idx' becomes 'idx' where idx is an integer
                main_df['batch_id'] = main_df['name'].map(lambda x: int(x.replace('SBatchConsumed_','')))
                #  drop "duration" column
                main_df = main_df.drop(columns=['name','duration'])
                continue

            if "worker_pid" in file:
                df = pd.read_csv(os.path.join(root, file)
                                , header=None)

                # add header
                df.columns = ['name','start_ts','duration']

                # names that start with 'SBatchPreprocessed'
                df = df[df['name'].str.startswith('SBatchPreprocessed')]
                # map 'SBatchPreprocessed_' such that 'SBatchPreprocessed_idx' becomes 'idx' where idx is an integer
                df['batch_id'] = df['name'].map(lambda x: int(x.replace('SBatchPreprocessed_','')))

                df['preprocess_finish_ts'] = df['start_ts'] + df['duration']

                # drop start_ts and duration columns
                df = df.drop(columns=['name','start_ts','duration']) 

                # concatentate all dataframes
                worker_df = pd.concat([worker_df, df])
        
        if worker_df.empty:
            continue

        batch = root.split('/')[-1].split('b')[1].split('_')[0] # <long_dir_path> -> 128_gpu4 -> 128

        label = root.split('/')[-1] # retrieves b128_gpu4 kind of label
        print(f'{label}:')

        # merge main_df and worker_df
        final_df = pd.merge(worker_df, main_df, on='batch_id')
        final_df['wait_time'] = final_df['consume_ts'] - final_df['preprocess_finish_ts']

        # divide by 1e6 to convert to ms
        final_df['wait_time'] = final_df['wait_time'] / 1000000

        if 'b512' in root:
            # if wait_time is greater than 80 ms, then it is a good data point, to remove last few outliers
            final_df = final_df[final_df['wait_time'] > 80]
        
        # print % of batches which had high wait time < 250 ms pretty print
        print (f'{100*len(final_df[final_df["wait_time"] < 250])/len(final_df):.2f}% of batches had wait time < 250 ms')


        if prev_batch is None:
            prev_batch = batch 
        elif batch != prev_batch:

            # plt.figure(figsize=fig_size)
            plt.gcf().set_size_inches(fig_size[0], fig_size[1])
            # label x axis
            plt.xlabel(f'Wait time for batch ids to be consumed (sorted by {sort_by}) (batch size = {prev_batch})', labelpad=40)
            # label y axis
            plt.ylabel('Wait time in ms',labelpad=40)
            plt.tick_params(axis='y', which='major', pad=40) 
            plt.tick_params(axis='x', which='major', pad=40) 
            # add legend to bottom right and increase size of legend markers
            # plt.legend(loc='lower right',markerscale=4)
            # plt.legend(bbox_to_anchor=(1.05, 1), markerscale=4)
            plt.tight_layout()
            fig_path = os.path.join(fig_dir,f'{fig_prefix}{prev_batch}_diff_consumed_preprocess_end_per_batch_{sort_by}.png')
            plt.savefig(fig_path)
            plt.clf()
            prev_batch = batch            
        
        mean = np.mean(final_df["wait_time"])
        std = np.std(final_df["wait_time"])
        print (f'avg = {mean:.2f} ms, std = {std:.2f} ({100*std/mean:.2f}% of avg) ms, min = {final_df["wait_time"].min():.2f} ms, max = {final_df["wait_time"].max():.2f} ms')
        # print % of batches which had high wait time > 500 ms pretty print
        print (f'{100*len(final_df[final_df["wait_time"] > 500])/len(final_df):.2f}% of batches had wait time > 500 ms')
        final_df = final_df.sort_values(by=[sort_by])
        if sort_by == 'batch_id':
            plt.scatter(final_df['batch_id'], final_df['wait_time'], label=label,s=300)
        else:
            # reset index
            final_df = final_df.reset_index(drop=True)
            p = np.array([0, 25, 50, 75, 100])
            plt.scatter(final_df.index, final_df['wait_time'], label=label, marker='s', s=240)

            # display grid lines and make it dashed and thick
            plt.grid(visible=True, which='major', linestyle='--', linewidth=10)
            # add a horizontal line at y = 500 ms thick
            plt.axhline(y=500, color='r', linestyle='-', linewidth=10)
            # add a label to the hosrizontal line
            plt.text(0, 550, ' 500 ms', fontsize=80, color='r')
            plt.xticks((len(final_df['wait_time'])-1) * p/100., map(str, p))
        # plot on log scale
        plt.yscale('log')

    # plt.figure(figsize=fig_size)
    plt.gcf().set_size_inches(fig_size[0], fig_size[1])
    # label x axis
    plt.xlabel(f'Wait time for batch ids to be consumed (sorted by {sort_by})', labelpad=40)
    # label y axis
    plt.ylabel('Wait time in ms',labelpad=40)
    plt.tick_params(axis='y', which='major', pad=40) 
    plt.tick_params(axis='x', which='major', pad=40) 
    # add legend to bottom right and increase size of legend markers
    # plt.legend(loc='lower right',markerscale=4)
    # plt.legend(bbox_to_anchor=(1.05, 1), markerscale=4)
    plt.tight_layout()
    fig_path = os.path.join(fig_dir,f'{fig_prefix}{prev_batch}_diff_consumed_preprocess_end_per_batch_{sort_by}.png')
    plt.savefig(fig_path)
    plt.clf()

# %%
# sort_by = 'batch_id' or 'duration'
def plotter_diff_consumed_wait_end_per_batch_time(target_dir,sort_by='batch_id',fig_size=(50,25),remove_outliers=True,fig_prefix='',fig_dir=''):
    
    root_to_files = {}
    for root, dirs, files in os.walk(target_dir):
        root_to_files[root] = files
    roots = sorted(root_to_files, key=lambda x: natsort.natsort_key(x.lower()))

    prev_batch = None
    for root in roots:
        if 'e2e' in root:
            continue
        print(root)
        files = root_to_files[root]
        main_df = pd.DataFrame()
        
        for file in files:
            if "main_pid" in file:
                main_df = pd.read_csv(os.path.join(root, file)
                            , header=None)

                # add header
                main_df.columns = ['name','start_ts','duration']

                # names that start with 'SBatchWait'
                wait_df = main_df[main_df['name'].str.startswith('SBatchWait')].copy()
                # map 'SBatchWait_' such that 'SBatchWait_idx' becomes 'idx' where idx is an integer
                wait_df['batch_id'] = wait_df['name'].map(lambda x: int(x.replace('SBatchWait_','')))
                wait_df['wait_end_ts'] = wait_df['start_ts'] + wait_df['duration']
                #  drop columns
                wait_df = wait_df.drop(columns=['name','start_ts','duration'])

                # names that start with 'SBatchConsumed'
                consumed_df = main_df[main_df['name'].str.startswith('SBatchConsumed')].copy()
                # map 'SBatchConsumed_' such that 'SBatchConsumed_idx' becomes 'idx' where idx is an integer
                consumed_df['batch_id'] = consumed_df['name'].map(lambda x: int(x.replace('SBatchConsumed_','')))
                consumed_df['consumed_ts'] = consumed_df['start_ts']
                #  drop columns
                consumed_df = consumed_df.drop(columns=['name','start_ts','duration'])     

                # merge wait_df and consumed_df
                main_df = pd.merge(wait_df, consumed_df, on='batch_id')           
        if main_df.empty:
            continue

        batch = root.split('/')[-1].split('b')[1].split('_')[0] # <long_dir_path> -> 128_gpu4 -> 128

        label = root.split('/')[-1] # retrieves b128_gpu4 kind of label
        print(f'{label}:')

        # merge main_df and worker_df
        main_df['wait_time'] = main_df['consumed_ts'] - main_df['wait_end_ts']

        # divide by 1e6 to convert to ms
        main_df['wait_time'] = main_df['wait_time'] / 1000000

        if prev_batch is None:
            prev_batch = batch 
        elif batch != prev_batch:

            # plt.figure(figsize=fig_size)
            plt.gcf().set_size_inches(fig_size[0], fig_size[1])
            # label x axis
            plt.xlabel(f'Wait time for batch ids to be consumed (sorted by {sort_by}) (batch size = {prev_batch})',labelpad=40)
            # label y axis
            plt.ylabel('Wait time in ms',labelpad=40)
            plt.tick_params(axis='y', which='major', pad=40) 
            plt.tick_params(axis='x', which='major', pad=40) 
            # add legend to bottom right and increase size of legend markers
            plt.legend(loc='lower right',markerscale=4)
            # plt.legend(bbox_to_anchor=(1.05, 1), markerscale=4)
            fig_path = os.path.join(fig_dir,f'{fig_prefix}{prev_batch}_diff_consumed_wait_end_per_batch.png')
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.clf()
            prev_batch = batch            
        
        mean = np.mean(main_df["wait_time"])
        std = np.std(main_df["wait_time"])
        print (f'avg = {mean:.2f} ms, std = {std:.2f} ({100*std/mean:.2f}% of avg) ms, min = {main_df["wait_time"].min():.2f} ms, max = {main_df["wait_time"].max():.2f} ms')
        # print % of batches which had high wait time > 100 ms pretty print
        print (f'{100*len(main_df[main_df["wait_time"] > 100])/len(main_df):.2f}% of batches had wait time > 100 ms')
        main_df = main_df.sort_values(by=[sort_by])
        plt.scatter(main_df['batch_id'], main_df['wait_time'], label=label,s=300)
        # plot on log scale
        plt.yscale('log')

    # plt.figure(figsize=fig_size)
    plt.gcf().set_size_inches(fig_size[0], fig_size[1])
    # label x axis
    plt.xlabel(f'Wait time for batch ids to be consumed (sorted by {sort_by})',labelpad=40)
    # label y axis
    plt.ylabel('Wait time in ms',labelpad=40)
    plt.tick_params(axis='y', which='major', pad=40) 
    plt.tick_params(axis='x', which='major', pad=40) 
    # add legend to bottom right and increase size of legend markers
    plt.legend(loc='lower right',markerscale=4)
    # plt.legend(bbox_to_anchor=(1.05, 1), markerscale=4)
    plt.tight_layout()
    fig_path = os.path.join(fig_dir,f'{fig_prefix}{prev_batch}_diff_consumed_wait_end_per_batch.png')
    plt.savefig(fig_path)
    plt.clf()

# # %%
print("preprocessing time")
plotter_preprocessing_time(args.data_dir,sort_by=args.sort_criteria,remove_outliers=True,fig_dir=args.fig_dir, fig_size=figsize)
print("----------------------------------------------\n")
    
# # %%
print("main process wait time")
plotter_preprocessing_wait_time(args.data_dir,fig_dir=args.fig_dir, fig_size=figsize)
print("----------------------------------------------\n")

# %%
print("[imagenet]  plotting difference between batch consumed and end of batch preprocessing")
plotter_diff_consumed_preprocess_end_per_batch_time(args.data_dir,remove_outliers=True,fig_dir=args.fig_dir, fig_size=figsize,sort_by=args.sort_criteria)
print("----------------------------------------------\n")

# # %%
print("[imagenet] plotting difference between batch consumed and end of batch wait")
plotter_diff_consumed_wait_end_per_batch_time(args.data_dir,remove_outliers=True, fig_dir=args.fig_dir, fig_size=figsize)
print("----------------------------------------------\n")