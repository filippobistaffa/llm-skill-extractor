import argparse as ap
import pandas as pd
import numpy as np
import json
import os
import re

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
oaclient = OpenAI()

def get_embedding(text, model='text-embedding-3-small'):
   text = text.replace('\n', ' ')
   return np.array(oaclient.embeddings.create(input=[text], model=model).data[0].embedding)

def cosine_similarity(a, b):
    if len(a) != len(b):
        raise ValueError(f'Arrays of different lengths: {len(a)} != {len(b)}')
    #cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_sim = np.dot(a, b) # assumes that np.linalg.norm(a) == np.linalg.norm(b) == 1 (holds in case of OpenAI embeddings)
    return cos_sim

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--description', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'description.txt'))
    parser.add_argument('--llm', type=str, default='gpt-4o-mini', choices=[
        # OpenAI
        'gpt-3.5-turbo',
        'gpt-4o',
        'gpt-4o-mini',
        # Mistral AI
        'open-mistral-7b',
        'open-mixtral-8x7b',
        'open-mixtral-8x22b',
        'mistral-small-latest',
        'mistral-large-latest',
    ])
    parser.add_argument('--embeddings', type=str, default='small', choices=['small', 'large'])
    parser.add_argument('--temperature', type=float, default=0) # should be 0 <= t <= 2 for OpenAI, 0 <= t <= 1 for Mistral AI
    parser.add_argument('-n', type=int, default=5)
    parser.add_argument('-k', type=int, default=5)
    parser.add_argument('--json', type=str)
    args, additional = parser.parse_known_args()

    # build prompt
    with open(args.description) as f:
        description = f'' + ''.join(f.readlines()).strip()
    prompt = f'Output a list of {args.n} specialist tasks taken from the Australian Skill Framework that are related to the following course:\n{description}'
    instructions = 'Answer only with a numbered list of skills, nothing else.'

    if args.llm.startswith('gpt'):
        # refer to https://platform.openai.com/docs/api-reference/chat
        response = oaclient.chat.completions.create(
            model=args.llm,
            temperature=args.temperature,
            messages=[
                {'role': 'user', 'content': prompt},
                {'role': 'system', 'content': instructions},
            ]
        ).choices[0].message.content
    else:
        from mistralai.client import MistralClient
        from mistralai.models.chat_completion import ChatMessage
        # refer to https://docs.mistral.ai/api/
        client = MistralClient(api_key=os.environ['MISTRAL_API_KEY'])
        response = client.chat(
            model=args.llm,
            temperature=args.temperature,
            messages=[
                ChatMessage(role='user', content=prompt),
                ChatMessage(role='system', content=instructions)
            ]
        ).choices[0].message.content

    # keep only lines of the numbered list
    response = [line.split('. ', 1)[1].strip() for line in response.splitlines() if re.match(r'^\d+\. ', line)]
    print(f'LLM ({args.llm}) response:\n- {"\n- ".join(response)}')

    # compute embeddings for the labels produced by the LLM
    labels = pd.DataFrame(response, columns=['label'])
    labels['embedding'] = labels['label'].apply(lambda x: get_embedding(x, model=f'text-embedding-3-{args.embeddings}'))

    # load precomputed embeddings of the skills in the framework
    embeddings = pd.read_pickle(f'skills-embeddings-3-{args.embeddings}.tar.gz')

    # dictionary to store results
    output = {
        'labels': dict()
    }

    # for each skill produced the LLM, compute the k skills in the framework with the highest cosine similarity
    for i, label in labels.iterrows():
        embeddings['cos_sim'] = embeddings['embedding'].apply(lambda x: cosine_similarity(x, label['embedding']))
        topn = embeddings.nlargest(args.k, 'cos_sim')
        print('{}{}'.format('\n' if i < args.n - 1 else '', label['label']))
        print(topn)
        output['labels'][label['label']] = {
            #'embedding': label['embedding'].tolist(),
            'skills': [
                {
                    'skill': top_skill['skill'],
                    #'embedding': top_skill['embedding'].tolist(),
                    'similarity': top_skill['cos_sim']
                } for _, top_skill in topn.iterrows()
            ]
        }

    if args.json is not None:
        with open(args.json, 'w') as f:
            json.dump(output, f, indent=2)
