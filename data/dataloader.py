import json
import os
import re

import pandas as pd
import spacy
from spacy.training import offsets_to_biluo_tags

import config

ENTITIES = ["FOOD", "QUANTITY", "UNIT", "PROCESS", "PHYSICAL_QUALITY", "COLOR",
            "TASTE", "PURPOSE", "PART"]
NLP = spacy.load("en_core_web_sm")


def _read_csv(dataset):
    data = pd.read_csv(
        os.path.join(config.DATA_PATH, f'{dataset}.csv')
    )
    return data


def _span_to_biluo(recipe, span_entities):
    """
    :param span_entities: list of span entities, eg. [(span_start, span_end,
    "FOOD"), (span_start, span_end, "PROCESS")]
    :return: list of BILUO entities, eg. ["O", "B-FOOD", "L-FOOD",
    "U-PROCESS"] along with tokenized recipe
    """
    doc = NLP(recipe.replace("\n", " "))
    tokenized_recipe = [token.text for token in doc]
    spans = offsets_to_biluo_tags(doc, span_entities)
    return tokenized_recipe, spans


def _biluo_to_iob(biluo_entities):
    """
    :param biluo_entities: list of BILUO entities, eg. ["O", "B-FOOD", "L-FOOD",
    "U-PROCESS"]
    :return: list of BIO entities, eg. ["O", "B-FOOD", "I-FOOD", "B-PROCESS"]
    """
    bio_entities = [entity.replace("L-", "I-").replace("U-", "B-")
                    for entity in biluo_entities]
    return bio_entities


def _span_to_iob(recipe, span_entities):
    """
    :param span_entities: list of span entities, eg. [(span_start, span_end,
    "FOOD"), (span_start, span_end, "PROCESS")]
    :return: list of BIO entities, eg. ["O", "B-FOOD", "I-FOOD", "B-PROCESS"]
    """
    tokenized_recipe, biluo_entities = _span_to_biluo(recipe, span_entities)
    bio_entities = _biluo_to_iob(biluo_entities)
    return tokenized_recipe, bio_entities


def prepare_data(df, entities_format="spans"):
    """
    df: TASTEset as pd.DataFrame or a path to the TASTEset
    entities_format: the format of entities. If equal to 'bio', entities
    will be of the following format: [[B-FOOD, I-FOOD, O, ...], [B-UNIT, ...]].
    If equal to span, entities will be of the following format:
    [[(0, 6, FOOD), (10, 15, PROCESS), ...], [(0, 2, UNIT), ...]]
    return:
    list of recipes and corresponding list of entities
    """

    all_recipes = df["ingredients"].to_list()
    all_entities = []

    for idx in df.index:
        ingredients_entities = json.loads(df.at[idx, "ingredients_entities"])
        entities = []

        for entity_dict in ingredients_entities:
            entities.append((entity_dict["start"], entity_dict["end"],
                             entity_dict["type"]))

        if entities_format == "IOB":
            tokenized_recipe, entities = _span_to_iob(all_recipes[idx],
                                                      entities)
            all_recipes[idx] = tokenized_recipe

        all_entities.append(entities)

    return all_recipes, all_entities


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

    post_df = post_df.reset_index(drop=True)
    pass


if __name__ == '__main__':
    _data_preprocessing()
