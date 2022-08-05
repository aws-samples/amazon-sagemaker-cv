import sys
import os
import subprocess
import json
from datetime import datetime

def unarchive_data(data_dir, target='coco.tar'):
    print("Unarchiving COCO data")
    os.system('tar -xf {0} -C {1}'.format(os.path.join(data_dir, target), data_dir))
    print(os.listdir(data_dir))
    
def get_training_world():

    """
    Calculates number of devices in Sagemaker distributed cluster
    """
    
    # Get params of Sagemaker distributed cluster from predefined env variables
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    num_cpus = int(os.environ["SM_NUM_CPUS"])
    hosts = json.loads(os.environ["SM_HOSTS"])
    current_host = os.environ["SM_CURRENT_HOST"]

    # Define PyTorch training world
    world = {}
    world["number_of_processes"] = num_gpus if num_gpus > 0 else num_cpus
    world["number_of_machines"] = len(hosts)
    world["size"] = world["number_of_processes"] * world["number_of_machines"]
    world["machine_rank"] = hosts.index(current_host)
    world["master_addr"] = hosts[0]
    world["master_port"] = "55555" # port is defined by Sagemaker

    return world

def main():
    # use tar from S3 so it loads faster
    data_dir="/opt/ml/input/data/all_data/"
    train_script="/opt/ml/code/train.py"
    unarchive_data(data_dir)
    world = get_training_world()
    sm_args = json.loads(os.environ["SM_HPS"])
    args = [f"--{key} {value}" for key, value in sm_args.items()]
    # Include EFA ENV variables
    # These get ignored on non-EFA instances
    launch_config = ["export FI_PROVIDER=efa &&",
                     "export FI_EFA_TX_MIN_CREDITS=64 &&",
                     "export NCCL_DEBUG=INFO &&"
                     "export NCCL_TREE_THRESHOLD=0 &&"
                     "export NCCL_SOCKET_IFNAME=eth0 &&"
                     "export FI_EFA_USE_DEVICE_RDMA=1 &&"
                     "torchrun",
                     f"--nnodes={world['number_of_machines']}",
                     f"--node_rank={world['machine_rank']}",
                     f"--nproc_per_node={world['number_of_processes']}",
                     f"--master_addr={world['master_addr']}",
                     f"--master_port={world['master_port']}",
                     train_script]
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
    
if __name__ == "__main__":
    main()