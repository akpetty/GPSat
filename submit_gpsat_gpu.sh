#!/bin/bash
# Run as: sbatch submit_gpsat_gpu.sh 2 gpu_test_1 True

#SBATCH --job-name=gpu_gpsat    # create a short name for your job
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1               # total number of tasks across all nodes
##SBATCH --gres=gpu:1                # node count
#SBATCH --partition=compute

##SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
##SBATCH --mem=80G                # memory per node (4G per cpu-core is default)
##SBATCH --time=2:00:00          # total run time limit (HH:MM:SS)

##SBATCH --mail-user=akpetty@umd.edu
##SBATCH --mail-type=ALL

##SBATCH --partition=compute                # node count
##SBATCH --mem-per-cpu=20G          # memory per CPU (need to pick which memory option to use)
##SBATCH --nodelist icesat202

export PATH="/home/aapetty/.conda/envs/is2_39_gp/bin:$PATH"
module load gcc/12.1.0
source activate is2_39_gp
python /explore/nobackup/people/aapetty/GitHub/GPSat/IS2_GPSat_train.py --num_days $1 --name $2 --sic $3
#echo $1
#echo $2
#echo $3
#python /home/aapetty/GitHub/ICESat-2-sea-ice-thickness-adapt/Code/gen_IS2SITDAT4.py --run_str "run_adapt_4" --start_date "2019-04" --end_date "2019-04" --beam "bnum3"
