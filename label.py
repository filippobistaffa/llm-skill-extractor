import argparse as ap
import pandas as pd
import numpy as np
import sys
import os


if __name__ == "__main__":

    llama_cpp_subdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'llama.cpp')

    # parse command-line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('--skills_dataset', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'skills.txt'))
    parser.add_argument('--description', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'description.txt'))
    parser.add_argument('--model', type=str, default=os.path.join(llama_cpp_subdir, 'models', 'vicuna-13b-v1.5-16k.Q4_K_M.gguf'))
    parser.add_argument('--format', type=str, default='USER: {}\nASSISTANT:')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cmd', type=str, default='cmd.sh')
    args, additional = parser.parse_known_args()

    # build prompt
    skills_list = pd.read_csv(args.skills_dataset, sep='\t', header=None).values.ravel()
    skills_string = '- ' + '\n- '.join(skills_list) + '\n'

    with open(args.description) as f:
        description = f'' + ''.join(f.readlines()).strip()

    prompt = f'Given the following list of {len(skills_list)} skills:\n{skills_string}Which of the above-mentioned skills could be acquired by participating to the following course:\n{description}'
    #prompt = f'Which of the specialist tasks in the Australian Skills Framework are most related to the following course:\n{description}'
    prompt_format = ('"' + args.format + '"').format(prompt).replace("\n", "\\n").replace("\t", "\\t")

    # llama.cpp parameters
    llama_cpp_params = {
        '--model': args.model,
        '--ctx-size': str(len(prompt_format)),
        '--seed': str(args.seed),
        '--prompt': prompt_format,
        '--repeat_penalty': '1.1',
        '--n-predict': '-1',
        '--temp': '0.7',
    }

    # build subprocess (llama.cpp) command-line
    command_line = [os.path.join(llama_cpp_subdir, 'build', 'bin' ,'main'), '--escape', '--log-disable']
    for (param, value) in llama_cpp_params.items():
        command_line.extend([param, value])
    command_line.extend(additional) # by putting additional at the end we can override the default ones

    with open(args.cmd, "w") as f:
        f.write(" ".join(command_line))
