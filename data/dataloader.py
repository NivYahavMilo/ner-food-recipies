"""
This script converts 'TASTEset' dataset original tagging structure into a
IOB tagging. saves it to csv in 'data' directory
"""

import json
import pandas as pd
from data_utils import _read_csv

def _data_preprocessing():
    data = _read_csv("TASTEset")
    post_df = pd.DataFrame()
    post_df['sample'] = 0
    iob_df = pd.DataFrame()
    jj = 0
    for ii, row in data.iterrows():
        entities = json.loads(row['ingredients_entities'])
        post_df = post_df.append(pd.DataFrame(entities))
        post_df['sample'][jj:jj + len(entities)] = ii + 1
        jj += len(entities)
        d = form_iob_tagging_format(post_df)
        dd = {}
        dd['entity'] = [tag[0] for ent, tag in d.items()]
        dd['tag'] = [tag[1] for ent, tag in d.items()]
        dd['sample'] = [ii + 1 for _ in range(len(dd['tag']))]
        iob_df = iob_df.append(pd.DataFrame.from_dict(dd), ignore_index=True)
    iob_df.to_csv("IOB tagging.csv")


def form_iob_tagging_format(df_section: pd.DataFrame):
    df_section = df_section.to_dict()
    iob_dict = {}
    jj = 0
    l = 0
    while bool(df_section['entity']):
        ent = df_section['entity'][jj]
        tag = df_section['type'][jj]
        if len(ent.split()) == 1:
            iob_dict[jj + l] = (ent, f'B-{tag}')
            l = 0
        else:
            ents = ent.split()
            for l, e in enumerate(ents):
                if l == 0:
                    iob_dict[jj] = (e, f'B-{tag}')
                else:
                    iob_dict[jj + l] = (e, f'I-{tag}')
        df_section['entity'].pop(jj)
        jj += 1
    return iob_dict


if __name__ == '__main__':
    _data_preprocessing()
