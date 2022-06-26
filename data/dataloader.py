import json
import os
import re
import config
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
ENTITIES = ["FOOD", "QUANTITY", "UNIT", "PROCESS", "PHYSICAL_QUALITY", "COLOR",
            "TASTE", "PURPOSE", "PART"]


def _read_csv(dataset):
    data = pd.read_csv(
        os.path.join(config.DATA_PATH, f'{dataset}.csv')
    )
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
                jj+=1
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
    jj = 0
    for ii, row in data.iterrows():

        recipe = row['ingredients']
        entities = json.loads(row['ingredients_entities'])

        post_df = post_df.append(pd.DataFrame(entities))

        post_df['sample'][jj:jj+len(entities)] = ii+1
        jj += len(entities)

        post_df = form_iob_tagging_format(post_df)

    post_df = post_df.reset_index(drop=True)
    pass

def form_iob_tagging_format(df_section: pd.DataFrame):
    for idx, row in df_section.iterrows():
        if len(row['entity'].split())==1:
            df_section.loc[idx, 'tag'] = f'B-{row["type"]}'
        else:
            df_section['entity'][idx] = row['entity'].split()
            df_section = df_section.explode(['entity'])
            pass

    return df_section





if __name__ == '__main__':
    _data_preprocessing()
