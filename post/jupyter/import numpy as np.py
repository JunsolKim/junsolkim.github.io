import numpy as np
import pandas as pd

def subreddit_benchmark(feature_file='changemyview_sent_vader.npy', threshold=10):
    # feature_file: file location of between-user features (e.g., sentiment)
    # threshold: number of interactions indicating the "affiliations" to subreddit

    # subreddit-user dataframe
    A = np.load('/project/jevans/hongkai/matrix_files/A/subreddit_user_count_pair.npy')
    df_A = pd.DataFrame(A, columns=['subreddit_id', 'user_id', 'count'])
    df_A = df_A.loc[df_A['count'] >= threshold].reset_index(drop=True)

    # user-user feature dataframe
    B = np.load(f'/project/jevans/hongkai/matrix_files/B/{feature_file}')
    df_B = pd.DataFrame(B, columns=['user_id_1', 'user_id_2', 'feature'])

    # df_merged1: average feature when user in a subreddit A interacts with user B
    df_merged1 = df_A.merge(df_B, left_on='user_id', right_on='user_id_1', how='right').rename({'subreddit_id': 'subreddit_id_1'}, axis=1)
    df_merged1 = df_merged1[['subreddit_id_1', 'user_id_2', 'feature']].groupby(['subreddit_id_1', 'user_id_2']).feature.mean().reset_index()

    # df_merged1: average feature when an user in a subreddit A interacts with an user in subreddit B
    df_merged2 = df_A.merge(df_merged1, left_on='user_id', right_on='user_id_2', how='right').rename({'subreddit_id': 'subreddit_id_2'}, axis=1)
    df_merged2 = df_merged2[['subreddit_id_1', 'subreddit_id_2', 'feature']].groupby(['subreddit_id_1', 'subreddit_id_2']).mean().reset_index()

    # columns: 'subreddit_id_1', 'subreddit_id_2', 'feature'
    return df_merged2

for i in [10, 50, 100]:
    for filename in ['changemyview_sent_vader.npy', 'changemyview_sent_change.npy', 'changemyview_count.npy', 'changemyview_sustainability.npy']:
        print(filename, i)
        subreddit_benchmark(f'{filename}.npy', 10).to_csv(f'/project/jevans/junsol/benchmark/{filename}_{i}.csv')

