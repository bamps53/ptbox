from pathlib import Path
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    print('split train dataset.')
    df = pd.read_csv('../input/bengaliai-cv19/train.csv')
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for i, (_, val_idx) in enumerate(kf.split(df, df['grapheme_root'])):
        df.loc[val_idx, 'folds'] = i
    df = df.reset_index()
    df = df.rename(columns={'index':'original_index'})

    le = LabelEncoder()
    df['combined'] = df['grapheme_root'].astype(str) + '_' + df['vowel_diacritic'].astype(str) + '_' + df['consonant_diacritic'].astype(str)
    df['unique_label'] = le.fit_transform(df['combined'])

    df.to_csv('data/folds.csv',index=False)

    g_num_per_class = df.grapheme_root.value_counts().sort_index().values
    v_num_per_class = df.vowel_diacritic.value_counts().sort_index().values
    c_num_per_class = df.consonant_diacritic.value_counts().sort_index().values
    a_num_per_class = df['unique_label'].value_counts().sort_index().values

    np.save('data/g_num_per_class', g_num_per_class)
    np.save('data/v_num_per_class', v_num_per_class)
    np.save('data/c_num_per_class', c_num_per_class)
    np.save('data/a_num_per_class', a_num_per_class)


