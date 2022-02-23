import sys 
import os
import subprocess

source = sys.argv[1]
target = sys.argv[2]
base_path = sys.argv[3]
other_params = sys.argv[4:]
other_params = ' '.join(other_params)

def run_command(command):
    string = f"# $ {command} #"
    sep = '#'*len(string)
    print("")
    print(sep)
    print(f"$ {command}")
    print(sep)
    process = subprocess.run(command.split(), stdout=subprocess.PIPE)
    numeric_output = process.stdout.decode('ascii').split('\n')[-2]
    try:
        auroc, acc_nr, acc_r, unk, hos = numeric_output.split(',')
        result = {
                "auroc": auroc,
                "acc_nr": acc_nr,
                "acc_r": acc_r,
                "unk": unk, 
                "hos": hos
                }
    except:
        print("There was an error while performing a test. Exit")
        return sys.exit(-1)

    return result

results = {}

# first episode
for episode in range(4):
    results[episode] = {}
    for do in range(5):
        command = f'python eval.py --source {source} --target {target} --load_path {base_path}_do_{do}/episode_{episode}.model --mode OWR_eval --eval_episode {episode} --dataorder {do} {other_params}'
        out = run_command(command)
        results[episode][do] = out

for key in ["auroc", "acc_nr", "acc_r", "unk", "hos"]:
    print(key)
    for episode in range(4):
        for do in range(4):
            print(results[episode][do][key], end=",")
        print(results[episode][4][key])
    print("")


