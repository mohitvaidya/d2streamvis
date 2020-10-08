import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.errors as errors
import azure.cosmos.http_constants as http_constants
import argparse
from pathlib import Path
import os
import json
from tqdm import tqdm

from ast import literal_eval
from os import listdir
from os.path import isfile, join
import concurrent.futures   



def push_files(file):
    COSMOS_ACCOUNT_URI="https://cosmos-ml.documents.azure.com:443"
    COSMOS_ACCOUNT_KEY="Xk2aRRmk45Ix6CJH72ZgzcbV0uQn4Ln2gYnAfdPY4gxi65X2odyA9BdIxlCWBkiWquodWSyHY7mFce1L5X9Nzg=="
    database_name = 'pipeline'

    client = cosmos_client.CosmosClient(url = COSMOS_ACCOUNT_URI, credential = COSMOS_ACCOUNT_KEY)

    database = client.get_database_client(database_name)
    container_name = 'custom_od'
    container = database.get_container_client(container_name)
    # basepath = f'./data'
    basepath=f'/app'


    with open(f'{basepath}/{file}_.json', 'rb') as f:
        data = json.load(f)
        container.upsert_item(data)
        print(f'Upload completed for {file}.json')

