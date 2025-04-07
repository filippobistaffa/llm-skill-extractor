from progress import test_progress
import argparse as ap
import pandas as pd
import numpy as np
import json
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
    parser.add_argument('--framework', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'framework.json'))
    parser.add_argument('--model', type=str, default='text-embedding-3-small', choices=['text-embedding-3-small', 'text-embedding-3-large'])
    parser.add_argument('--embeddings', type=str, default='skills-embeddings-3-small.tar.gz')
    args, additional = parser.parse_known_args()
    with open(args.framework) as f:
        framework = json.load(f)
    df = pd.DataFrame([skill['name'] for skill in framework], columns=['skill'])
    with test_progress as progress:
        task = progress.add_task("Generating embeddings...", total=len(df))
        embeddings = []
        for skill in df['skill']:
            embedding = get_embedding(skill, model=args.model)
            embeddings.append(embedding)
            progress.update(task, advance=1)
    df['embedding'] = embeddings
    print(f'Saving {args.embeddings}...')
    df.to_pickle(args.embeddings)
