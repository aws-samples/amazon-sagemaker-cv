import os
from subprocess import call

def get_data_tar(s3_path, file_name='data.tar', local_path='/opt/ml/input/data/all_data'):
    call(["aws", "s3", "cp", s3_path, os.path.join(local_path, file_name)])
    call(["tar", "-xf", os.path.join(local_path, file_name), '-C', local_path])
