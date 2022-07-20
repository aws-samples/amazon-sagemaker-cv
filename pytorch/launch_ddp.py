import sys
import os
import subprocess
import json
from utils import get_training_world, unarchive_data
from datetime import datetime

if __name__ == "__main__":
    start_time = datetime.now()
    print('Starting training...')
    # print("Training started at {}".format(start_time))
    world = get_training_world()
    sm_args = json.loads(os.environ["SM_HPS"])
    # if "unarchive" in sm_args:
    #     data_dir = sm_args.pop("unarchive")
    #     unarchive_data(data_dir)
    #     print("Unarchive completed in {}".format(datetime.now() - start_time))
    args = [f"--{key} {value}" for key, value in sm_args.items()]
    launch_config = ["python -m torch.distributed.launch", 
                     "--nnodes", str(world['number_of_machines']), 
                     "--node_rank", str(world['machine_rank']),
                     "--nproc_per_node", str(world['number_of_processes']), 
                     "--master_addr", world['master_addr'], 
                     "--master_port", world['master_port'],
                     "/opt/ml/code/train.py"]
    launch_config.extend(args)
    joint_cmd = " ".join(str(x) for x in launch_config)
    print("Following command will be executed: \n", joint_cmd)
    
    process = subprocess.Popen(joint_cmd,  stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    
    while True:
        output = process.stdout.readline()
        
        if process.poll() is not None:
            break
        if output:
            print(output.decode("utf-8").strip())
    rc = process.poll()
    
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=joint_cmd)
    
    sys.exit(process.returncode)
