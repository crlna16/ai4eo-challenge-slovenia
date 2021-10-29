## Challenge: the technical part

Documentation for using singularity on mistral:
https://www.dkrz.de/up/systems/mistral/singularity

### Create the docker image

As explained in the documentation, it is not possible to directly create the image directly on mistral as it requires sudo. I created the docker image on my laptop (ubuntu 20.04 running via wsl 2)


1. Start the docker daemon: `sudo dockerd`
2. Checkout the git repository: `git checkout git@gitlab.dkrz.de:k202141/ai4eo-challenge.git`
3. `cd ai4eo-challenge`
4. Create the docker image: `sudo docker build -f Dockerfile-jupyter-user-g-stripped-cuda11 -t ai4eo .`

### Transfer the image to mistral

You now have a local docker image. To transfer it to mistral, I used dockerhub: `hub.docker.com`. Register a user account (`$USER_DOCKERHUB`) and do the following:

1. `sudo docker tag ai4eo $USER_DOCKERHUB/ai4eo` 
2. `sudo docker push $USER_DOCKERHUB/ai4eo`

Login to mistral. Checkout the git repository there as well:

1. Checkout the git repository: `git@gitlab.dkrz.de:k202141/ai4eo-challenge.git`
2. `cd ai4eo-challenge`

Create the singularity image:

1. Start an interactive session on any node with internet access (GPUs on mistral / trial)
2. Activate singularity module: `module load singularity`
3. Pull the image `singularity pull docker://$USER_DOCKERHUB/ai4eo`

## Run the singularity container

Create an allocation for an interactive job on any of the amd nodes

1. `ssh trial.dkrz.de`
2. `salloc --partition=amd --time=04:00:00 --exclusive -A ka1176`
3. `ssh vader{N}` (use `squeue` to see where your interactive job is running)
4. Start the singularity container: `singularity shell --nv --bind /scratch/k/$USER/singularity/cache:/home/jovyan/.cache --bind /mnt/lustre02/work/ka1176/:/swork /work/ka1176/caroline/gitlab/ai4eo-challenge/ai4eo_latest.sif`

Options in detail: `--nv` for nvidia GPUs, bind cache to a writable folder (I used scratch space), bind shared data so it is visible when you work in the container. The `/work/ka1176/` folder is now available at `/swork/`.

Now we need to activate the conda environment:

1. `. "/opt/conda/etc/profile.d/conda.sh"`
2. `conda env list`
3. `conda activate eurodatacube-gpu-`


## Run the model 

Run the model with the appropriate input parameters, one example:

`python model.py --processed-data-dir /swork/shared_data/2021-ai4eo/dev_data/default/ --n-s2 1 --max-epochs 50 --learning-rate 1e-4 --batch-size 32 --scaling-factor 4 --large-kernel-size 9 --small-kernel-size 3 --patience 50  --n-channels 32 --input-channels 4 --bands B02 B03 B04`

The best model is saved.

## Create the submission file

Update the parameters in `create_submission.sh` to the ones you used to train the model in the step before (sorry for the inconvenience). Run the script

`sh create_submission.sh $SUBMISSION_FOLDER`

Use the notebook `submission.ipynb` to visualize your submission.

## Upload the submission

Login to the AI4EO challenge website and upload the `.tar.gz` folder. Shortly, the result will appear on the leaderboard.

`DONE!`

## Alternative: submit the job

Often it will make sense to submit training the model as a slurm job, rather than running an interactive session. Below is a demo script for such a submission. Change the variables `gitdir_c`, `scriptdir_c` for your corresponding paths, and adapt the arguments to `model.py` as needed.

```bash
#!/bin/bash
#SBATCH -p amd
#SBATCH -A ka1176
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=08:00:00

hostname

module load /sw/spack-amd/spack/modules/linux-centos8-zen2/singularity/3.7.0-gcc-10.2.0

# we will bind the folder /work/ka1176 to /swork in the singularity container
gitdir_c=/swork/caroline/gitlab/ai4eo-challenge/ai4eo  # gitlab dir (change this to gitlab directory as it would appear in the container)
scriptdir_c=/swork/caroline/jobs/ai4eo/tests           # script dir (change this to current directory as it would appear in the container)

# create run script for the job
echo "echo 'HELLO BOX'" > singularity_run.sh
echo "gitdir=$gitdir_c" >> singularity_run.sh
echo ". /opt/conda/etc/profile.d/conda.sh" >> singularity_run.sh
echo "conda activate eurodatacube-gpu-" >> singularity_run.sh
echo "echo \$gitdir" >> singularity_run.sh
echo "python \$gitdir/model.py --processed-data-dir /swork/shared_data/2021-ai4eo/dev_data/default/ --n-s2 225 --bands B02 B03 B04 --max-epochs 1 --input_channels 4 --learning-rate 1e-4 --patience 1000000" >> singularity_run.sh

# execute the singularity container
singularity exec --nv --bind /scratch/k/k202141/singularity/cache:/home/jovyan/.cache --bind /mnt/lustre02/work/ka1176/:/swork /work/ka1176/caroline/gitlab/ai4eo-challenge/ai4eo2_latest.sif bash $scriptdir_c/singularity_run.sh
```

## Submit a NNI job

Analogue to the previous script.
Below is a demo script for a submission using NNI. Change the variables `gitdir_c`, `scriptdir_c` for your corresponding paths, and adapt the arguments to `model.py` as needed.

```bash
#!/bin/bash
#SBATCH --partition=amd        # Specify partition name
#SBATCH --nodes=1              # Specify number of nodes
#SBATCH --mem=0                # Use entire memory of node
#SBATCH --exclusive            # Do not share node
#SBATCH --time=12:00:00        # Set a limit on the total run time
#SBATCH --mail-type=FAIL       # Notify user by email in case of job failure
#SBATCH --account=k20200

hostname

module load /sw/spack-amd/spack/modules/linux-centos8-zen2/singularity/3.7.0-gcc-10.2.0

gitdir_c=/swork/frauke/ai4eo-challenge/nni  # gitlab dir (change this to gitlab directory as it would appear in the container)
scriptdir_c=/swork/frauke/ai4eo-challenge/jobs # script dir (change this to current directory as it would appear in the container)


echo "echo 'HELLO BOX'" > singularity_run_nni.sh
echo "gitdir=$gitdir_c" >> singularity_run_nni.sh
echo ". '/opt/conda/etc/profile.d/conda.sh'" >> singularity_run_nni.sh
echo "conda activate eurodatacube-gpu-" >> singularity_run_nni.sh
echo "echo \$gitdir" >> singularity_run_nni.sh
echo "port=$((8080 + $RANDOM % 10000))" >> singularity_run_nni.sh
echo "nnictl create -c \$gitdir/config.yml --port \$port|| nnictl create -c \$gitdir/config.yml --port \$port|| nnictl create -c \$gitdir/config.yml --port \$port|| nnictl create -c \$gitdir/config.yml --port \$port" >> singularity_run_nni.sh
echo "sleep 11h 50m" >> singularity_run_nni.sh
echo "nnictl stop" >> singularity_run_nni.sh

# execute the singularity container
singularity exec --nv --bind /scratch/k/k202143/singularity/cache:/home/jovyan/.cache --bind /mnt/lustre02/work/ka1176/frauke/ai4eo-challenge/nni/:/opt/conda/envs/eurodatacube-gpu-/nni --bind /mnt/lustre02/work/ka1176/:/swork /work/ka1176/caroline/gitlab/ai4eo-challenge/ai4eo2_latest.sif bash $scriptdir_c/singularity_run_nni.sh
```

## Run predictions within a Jupyter notebook

This is required for the AI4EO challenge. Check the folder `final_submission` for the final notebook and the best model. 

The following has been developed locally on Ubuntu 20.04 (WSL2 on Windows 10). 

Build the final Docker image:

`sudo docker build -f Dockerfile-final -t eagleeyes_final .`

Run the Docker container:

`sudo docker run -p 8888:8888 -v PATH_TO_TEST_DATA_ON_HOST:/test_data/ eagleeyes_final`

You should then be able to open a Jupyter notebook at `127.0.0.1:8888` with the token provided. 
