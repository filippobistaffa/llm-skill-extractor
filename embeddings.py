import argparse as ap
import pandas as pd
import numpy as np
import os

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI()

def get_embedding(text, model='text-embedding-3-small'):
   text = text.replace('\n', ' ')
   return np.array(client.embeddings.create(input=[text], model=model).data[0].embedding)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('--skills', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'skills.csv'))
    parser.add_argument('--model', type=str, default='text-embedding-3-small', choices=['text-embedding-3-small', 'text-embedding-3-large'])
    parser.add_argument('--embeddings', type=str, default='skills-embeddings-3-small.tar.gz')
    args, additional = parser.parse_known_args()
    df = pd.read_csv(args.skills, index_col=0)
    df['embedding'] = df['skill'].apply(lambda x: get_embedding(x, model=args.model))
    df.to_pickle(args.embeddings)
