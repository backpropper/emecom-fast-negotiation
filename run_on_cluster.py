import os
import sys
import time

code_directory = '/home/guptaabh/scratch/emecom-fast-negotiation'
directory = code_directory

slurm_logs = os.path.join(directory, "slurm_logs")
slurm_scripts = os.path.join(directory, "slurm_scripts")
model_dir = os.path.join(directory, "models")
log_dir = os.path.join(directory, 'logs')

if not os.path.exists(slurm_logs):
    os.makedirs(slurm_logs)
if not os.path.exists(slurm_scripts):
    os.makedirs(slurm_scripts)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def _run_exp(batch, file_name='ecn.py', job_time="72:00:00"):
    save_str = batch['save_str']
    if batch['save_data']:
        batch['model_file'] = os.path.join(model_dir, save_str)
        batch['log_file'] = os.path.join(log_dir, save_str)
        if not os.path.exists(batch['model_file']):
            os.makedirs(batch['model_file'])
        if not os.path.exists(batch['log_file']):
            os.makedirs(batch['log_file'])

    jobcommand = f'python {file_name}'
    for key, value in batch.items():
        if key in ['save_data', 'save_str']:
            continue
        elif type(value) == bool:
            if value:
                jobcommand += f' --{key}'
        else:
            jobcommand += f' --{key} {value}'
    print(jobcommand)

    slurmfile = os.path.join(slurm_scripts, save_str + '.sh')
    with open(slurmfile, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --job-name={save_str}\n")
        f.write(f'#SBATCH --output={os.path.join(slurm_logs, save_str + ".out")}\n')
        # f.write(f'#SBATCH --error={os.path.join(slurm_logs, save_str + ".err")}\n')
        f.write("module load cuda cudnn\n")
        f.write(f"cd {code_directory}\n")
        f.write(jobcommand + "\n")

    s = f"sbatch --gres=gpu:1 --mem=20G -c4 --time={job_time} "
    s += f'{slurmfile} &'
    os.system(s)
    time.sleep(1)


job = {'no_load': True, 'render_every_seconds': 300, 'save_model': True}
for rs in [100, 105, 110, 115, 120]:
    for corr in [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]:
        job['seed'] = rs
        job['corr_utt_perc'] = corr
        job['save_str'] = f"seed{job['seed']}_corr{str(job['corr'])[-1]}"
        _run_exp(job)
