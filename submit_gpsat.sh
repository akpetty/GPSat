#!/bin/bash
# Run as: sbatch submit_gpsat.sh 15 baseline True 2019-04-15

#SBATCH --job-name=gpsat    # create a short name for your job
#SBATCH --nodes=1

#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=20G                # memory per node (4G per cpu-core is default)
#SBATCH --time=4:00:00          # total run time limit (HH:MM:SS)

##SBATCH --exclusive
##SBATCH --mail-user=akpetty@umd.edu
##SBATCH --mail-type=ALL

##SBATCH --partition=compute                # node count
##SBATCH --mem-per-cpu=20G          # memory per CPU (need to pick which memory option to use)
#SBATCH --nodelist icesat208

# This worked once i ssh into icesat201
export PATH="/home/aapetty/.conda/envs/is2_39_gp/bin:$PATH"
module load gcc/12.1.0
source activate is2_39_gp

#conda activate is2_39_gp
#conda install pandas --force-reinstall -y

#python /explore/nobackup/people/aapetty/GitHub/GPSat/IS2_GPSat_train.py --num_days $1 --name $2 --sic $3
python /explore/nobackup/people/aapetty/GitHub/GPSat/IS2_SMAP_GPSat_train.py --num_days $1 --name $2 --smap $3 --target_date $4
#echo $1
#echo $2
#echo $3
#python /home/aapetty/GitHub/ICESat-2-sea-ice-thickness-adapt/Code/gen_IS2SITDAT4.py --run_str "run_adapt_4" --start_date "2019-04" --end_date "2019-04" --beam "bnum3"
