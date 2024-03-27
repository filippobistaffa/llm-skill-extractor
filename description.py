import argparse as ap
import pandas as pd
import numpy as np
import llama_cpp
import os


if __name__ == "__main__":

    # parse command-line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('--skills_dataset', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'skills.txt'))
    parser.add_argument('--n_skills', type=int, default=3)
    parser.add_argument('--model', type=str, default=os.path.join('models', 'mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf'))
    parser.add_argument('--seed', type=int, default=llama_cpp.LLAMA_DEFAULT_SEED)
    parser.add_argument('--ctx', type=int, default=2048)
    parser.add_argument('--threads', type=int)
    parser.add_argument('--gpu-layers', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    args, additional = parser.parse_known_args()

    # build prompt
    np.random.seed(args.seed)
    skills_list = pd.read_csv(args.skills_dataset, sep='\t', header=None).values.ravel()
    skills_sample = np.random.choice(skills_list, size=args.n_skills)
    skills_string = ', '.join(skills_sample[:-1]).lower() + ', and ' + skills_sample[-1].lower()
    prompt = f'Give me the description of a professional course to learn how to {skills_string}.'

    # build LLM
    llm = llama_cpp.Llama(
        model_path = args.model,
        n_threads = args.threads,
        seed = args.seed,
        n_ctx = args.ctx,
        verbose = args.verbose,
        n_gpu_layers = args.gpu_layers
    )

    # generate and print output
    output = llm(prompt, max_tokens=-1, stop=['</s>'])
    print(output['choices'][0]['text'])
