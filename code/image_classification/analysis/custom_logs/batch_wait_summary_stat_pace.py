# %%
import os,pandas as pd, argparse, natsort

parser = argparse.ArgumentParser(description='batch_wait_summary_stat')
parser.add_argument('--data_dir', type=str,\
                     default='/users/rajveerb/pytorch_custom_log_one_epoch_imagenet_dataset/',\
                          help='directory where custom_log files are stored')
args = parser.parse_args()

def generate_batch_wait_stat(data_dir):
    root_to_result = {}

    root_to_files = {}
    for root, dirs, files in os.walk(data_dir):
        root_to_files[root] = files
    roots = sorted(root_to_files, key=lambda x: natsort.natsort_key(x.lower()))
    
    # recursively find all the log files
    for root in roots:
        if 'e2e' in root:
            continue
        files = root_to_files[root]
        merged_df = pd.DataFrame()
        combine_preprocess_df = pd.DataFrame()
        for file in files:
            # only process files end with .txt
            if file.startswith('custom_log'):
                    # read as a pandas dataframe
                    df = pd.read_csv(os.path.join(root, file),header=None)
                    df.columns = ['name', 'start', 'duration']
                    if 'main_pid' in file:
                        
                        # keep all rows in column 'name' with value starting with "SBatchWait" or "SBatchConsumed"
                        wait_df = df[df['name'].str.startswith('SBatchWait')].copy()
                        consumed_df = df[df['name'].str.startswith('SBatchConsumed')].copy()
                        # map SBatchWait to idx
                        wait_df['batch_id'] = wait_df['name'].map(lambda x: int(x.replace('SBatchWait_','')))
                        # map SBatchConsumed to idx
                        consumed_df['batch_id'] = consumed_df['name'].map(lambda x: int(x.replace('SBatchConsumed_','')))
                        wait_df['end_wait'] = wait_df['start'] + wait_df['duration']
                        wait_df['start_wait'] = wait_df['start']
                        consumed_df['start_consumed'] = consumed_df['start']
                        # drop duration column for wait_df
                        wait_df.drop(columns=['name','duration','start'],inplace=True)
                        # drop duration column for consumed_df
                        consumed_df.drop(columns=['name','duration','start'],inplace=True) 

                    elif 'worker_pid' in file:
                    
                        # keep all rows in column 'name' with value starting with "SBatchWait" or "SBatchConsumed"
                        preprocess_df = df[df['name'].str.startswith('SBatchPreprocessed')].copy()
                        # map SBatchWait to idx
                        preprocess_df['batch_id'] = preprocess_df['name'].map(lambda x: int(x.replace('SBatchPreprocessed_','')))
                        preprocess_df['end_preprocessed'] = preprocess_df['start'] + preprocess_df['duration']
                        # drop start and duration columns
                        preprocess_df.drop(columns=['name','start','duration'],inplace=True)
                        # concatenate preprocess_df
                        combine_preprocess_df = pd.concat([combine_preprocess_df,preprocess_df])
                    else:
                        continue
        # if merged_df is empty then continue
        if combine_preprocess_df.empty:
            continue
        # merge dfs with same idx
        merged_df = consumed_df.merge(wait_df, on='batch_id').merge(combine_preprocess_df, on='batch_id')
        
        conditions_met = (merged_df['end_preprocessed'] < merged_df['start_wait'])
                        # below is always met its an invariant
                    #  (merged_df['end_wait'] < merged_df['start_consumed']) )
        
        root_to_result[root] = [merged_df,conditions_met]
    
    return root_to_result

def print_batch_wait_stat(root_to_result):
    for root in root_to_result:
        merged_df,conditions_met = root_to_result[root]
        print('\n-------------------')
        print('root: ',root)
        # print('batches where end_preprocessed < start_wait i.e batch was ready to be consumed')
        # print(merged_df[conditions_met]['batch_id'].values)
        # print('batches where end_preprocessed > start_wait i.e batch was not ready to be consumed')
        # print(merged_df[~conditions_met]['batch_id'].values)
        print('number of batches where end_preprocessed < start_wait i.e batch was ready to be consumed')
        print(len(merged_df[conditions_met]))
        print('number of batches where end_preprocessed > start_wait i.e batch was not ready to be consumed')
        print(len(merged_df[~conditions_met]))
        print('% of batches where end_preprocessed < start_wait i.e batch was ready to be consumed')
        print(len(merged_df[conditions_met])/len(merged_df)*100)
        print('% of batches where end_preprocessed > start_wait i.e batch was not ready to be consumed')
        print(len(merged_df[~conditions_met])/len(merged_df)*100)


result = generate_batch_wait_stat(args.data_dir)

print_batch_wait_stat(result)