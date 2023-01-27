#!/bin/bash
#SBATCH -n 4                                    # Number of cores
#SBATCH -c 1
#SBATCH -N 1                                    # Ensure that all cores are on one machine
#SBATCH -t 0-5:00                             # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p xlin,xlin-lab,serial_requeue,shared    # Partition to submit to
#SBATCH -o heels_ld_decomp_%A_%a.out
#SBATCH -e heels_ld_decomp_%A_%a.err
#SBATCH --mem=100G                              # Memory pool for all cores
#SBATCH --mail-type=ALL
approx_method=$1

module load Anaconda3/5.0.1-fasrc02
source activate pcgc

cd ../
mkdir -p ./demo/input
mkdir -p ./demo/output

#-------------------------------------------------------
# run LD approximation algorithm 
#-------------------------------------------------------

python run_HEELS.py \
    --output_fp ./demo/input/ukb_30k_chr22_${approx_method} \
    --ld_fp ./data/ukb_30k_chr22_LD.npz \
    --ld_snp_fp ./data/ukb_30k_chr22_maf01_geno1 \
    --sumstats_fp ./data/ukb_30k_simul_pheno.txt \
    --N 30000 \
    --tol 1e-4 \
    --LD_approx_mode "manual" \
    --LD_approx_method ${approx_method} \
    --LD_approx_B 400\
    --LD_approx_R 300 \
    --constrain_sigma \
    --CV_metric "bias" \
    --LD_approx_num_valid_exp 10 \
    --LD_approx_valid_bias_tol 0.1 \
    --std_geno_fp ./data/ukb_30k_chr22_std_geno.npz \
    --stream-stdout
