import argparse as ap
import pandas as pd
import numpy as np
import llama_cpp
import os


if __name__ == "__main__":

    # parse command-line arguments
    parser = ap.ArgumentParser()
    group_io = parser.add_argument_group('input/output arguments')
    group_model = parser.add_argument_group('model arguments')
    group_paral = parser.add_argument_group('parallelization arguments')
    group_io.add_argument('--description', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'description.txt'))
    group_io.add_argument('--verbose', action='store_true')
    group_model.add_argument('--model', type=str, default=os.path.join('models', 'mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf'))
    group_model.add_argument('--seed', type=int, default=0)
    group_model.add_argument('--ctx', type=int, default=2048)
    group_paral.add_argument('--threads', type=int)
    group_paral.add_argument('--threads-batch', type=int)
    group_paral.add_argument('--gpu-layers', type=int, default=0)
    args, additional = parser.parse_known_args()

    # build prompt
    with open(args.description) as f:
        description = f'' + ''.join(f.readlines()).strip()
    prompt = f'Which of the specialist tasks in the Australian Skills Framework are most related to the following course?\n{description}'

    # format prompt according to the model
    model_formats = {
        'mixtral': '[INST] {} [/INST]',
        'vicuna': 'USER: {}\nASSISTANT:',
        'gemma': '<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model',
    }
    for model, model_format in model_formats.items():
        if model in args.model:
            prompt_format = model_format.format(prompt)

    # build LLM
    llm = llama_cpp.Llama(
        model_path = args.model,
        n_threads = args.threads,
        n_threads_batch = args.threads_batch,
        seed = args.seed,
        n_ctx = args.ctx,
        verbose = args.verbose,
        n_gpu_layers = args.gpu_layers,
        chat_format = ''
    )

    # generate and print output
    output = llm(prompt_format, max_tokens=-1)
    print(output['choices'][0]['text'])
