import json
import os
import re

import pandas as pd

import config

pd.options.mode.chained_assignment = None  # default='warn'
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
ENTITIES = ["FOOD", "QUANTITY", "UNIT", "PROCESS", "PHYSICAL_QUALITY", "COLOR",
            "TASTE", "PURPOSE", "PART"]


def _read_csv(dataset, i_col=None):
    data = pd.read_csv(
        os.path.join(config.DATA_PATH, f'{dataset}.csv'),
        index_col=i_col)
    return data


def preprocess_data():
    data = _read_csv("TASTEset")
    all_recipes = data["ingredients"].to_list()
    post_df = pd.DataFrame()

    for ii in data.index:
        ingredients_entities = json.loads(data.at[ii, "ingredients_entities"])
        recipes_tokens = re.split(r'\s+', data.at[ii, "ingredients"])
        jj = 0
        for token_dict in recipes_tokens:
            entity_dict = ingredients_entities[jj]
            if entity_dict['entity'] == recipes_tokens[jj]:
                post_df = post_df.append(pd.DataFrame(
                    {
                        'tag': entity_dict['type'],
                        'word': entity_dict['entity'],
                        'sample': ii
                    }, index=[jj]))
                jj += 1
            elif entity_dict['start'] - entity_dict['end'] == \
                    ingredients_entities[jj]['start'] - \
                    ingredients_entities[jj]['end']:
                post_df = post_df.append(pd.DataFrame({
                    'tag': ingredients_entities[jj]['type'],
                    'word': ingredients_entities[jj]['entity'],
                    'sample': ii
                }, index=[jj]))

                jj += len(ingredients_entities[jj]['entity'].split())

            else:
                post_df = post_df.append(pd.DataFrame({
                    'tag': 'O',
                    'word': recipes_tokens[jj]
                }))
                jj += len(recipes_tokens[jj].split()) - 1

        pass


def _data_preprocessing():
    data = _read_csv("TASTEset")
    post_df = pd.DataFrame()
    post_df['sample'] = 0
    iob_df = pd.DataFrame()
    jj = 0
    for ii, row in data.iterrows():
        recipe = row['ingredients']
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
