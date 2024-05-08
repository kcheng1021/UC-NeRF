import subprocess
from pathlib import Path

import gin


@gin.configurable()
class submitter():
    def __init__(self,
        name="test",
        log_dir=None,
        account=None,
        mail_user=None,
        mail_type="FAIL",
        nodes=1,
        ntasks=1,
        cpus_per_task=4,
        mem_mb=16,
        time="07-00:00:00",
        gres="gpu:rtx_3090:2",
    ):
        self.name = name
        self.log_dir = log_dir
        self.account = account
        self.mail_user = mail_user
        self.mail_type = mail_type
        self.nodes = nodes
        self.ntasks = ntasks
        self.cpus_per_task = cpus_per_task
        self.mem_mb = mem_mb
        self.time = time
        self.gres = gres

    
    def submit(self, command):
        Path(self.log_dir).mkdir(exist_ok=True, parents=True)
        sh_file = open(f"{self.log_dir}/{self.name}.sh", "w")
        sh_file.write(f'''#!/bin/bash

#SBATCH --job-name={self.name}
#SBATCH --output={self.log_dir}/{self.name}.out
#SBATCH --account={self.account}

#SBATCH --mail-user={self.mail_user}
#SBATCH --mail-type={self.mail_type}
#SBATCH --nodes={self.nodes}
#SBATCH --ntasks={self.ntasks}
#SBATCH --cpus-per-task={self.cpus_per_task}
#SBATCH --mem={1024 * self.mem_mb}

#SBATCH --gres={self.gres}
#SBATCH --time={self.time}


{command}

''')
        sh_file.close()
        subprocess.call(f"sbatch {self.log_dir}/{self.name}.sh", shell=True)


