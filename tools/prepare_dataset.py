"""
In this file, we will generate the following files:
* action_labels.csv: containing the action labels
* animal_labels.csv: containing the animal labels
* train.csv: containing train data entries with associated action & animal labels.
* val.csv: containing test data entries with associated action & animal labels.
"""

import numpy as np
import pandas as pd

metadata_file = './etc/AR_metadata.xlsx'
metadata = pd.ExcelFile(metadata_file)

# action_labels.csv
action = pd.read_excel(metadata, 'Action')
action_labels = action.loc[:, ['Label', 'Action']] \
    .rename(columns={'Label': 'i', 'Action': 'name'}) \
    .sort_values(by='i')
action_labels.to_csv('etc/action_labels.csv', index=False)

# animal_labels.csv
animal = pd.read_excel(metadata, 'Animal')
category_col = 'Sub-Class'
animal_labels = pd.DataFrame(
    animal[category_col].unique(), columns=['name']
    ) \
        .sort_values(by='name') \
        .reset_index(drop=True)
animal_labels_modified = animal_labels.copy()

# Only for Sub-Class
animal_labels_modified['name'] = animal_labels_modified['name'].map(lambda x: x.split(' / ')[0])

animal_labels_modified.to_csv('etc/animal_labels_107.csv', index_label='i')

animal_mapping = animal.loc[:, ['Animal', category_col]] \
    .join(
        animal_labels.reset_index().set_index('name'),
        on=category_col
    ).set_index('Animal')['index']
animal_mapping.index = animal_mapping.index.str.lower()

# train.csv & val.csv
replacement_animal = {
    "common quail": "common quail bird",
    "cuckoo bird": "common cuckoo bird"
}

def get_animal_list(s: str) -> str:
    l = eval(s)
    res = set()
    for item in l:
        if type(item) == str:
            # to handle weird data due to "'s"
            resi = eval(item)[0][0].replace('\'S', '\'s')
        else:
            resi = item[0]
        
        # replace with known valid replacement
        resi = resi.lower()
        resi = replacement_animal.get(resi, resi)
        res.add(str(animal_mapping[resi]))
    return ','.join(res)

ar = pd.read_excel(metadata_file, 'AR')
ar['animal_labels'] = ar['list_animal_action'].map(get_animal_list)
dataset = ar.loc[:, ['video_id','type','labels','animal_labels']]
dataset['video_id'] = dataset['video_id'].map(lambda x: x + '.mp4')

dataset[dataset['type']=='train'].to_csv('etc/train.csv', columns=['video_id','animal_labels','labels'], index=False, header=False, sep=' ')
dataset[dataset['type']=='test'].to_csv('etc/val.csv', columns=['video_id','animal_labels','labels'], index=False, header=False, sep=' ')