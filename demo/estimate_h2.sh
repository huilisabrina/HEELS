#!/bin/bash
#SBATCH -n 1                                    # Number of cores
#SBATCH -c 1
#SBATCH -N 1                                    # Ensure that all cores are on one machine
#SBATCH -t 0-1:00                             # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p test,xlin,xlin-lab,serial_requeue,shared    # Partition to submit to
#SBATCH -o heels_exact_ld_%A_%a.out
#SBATCH -e heels_exact_ld_%A_%a.err
#SBATCH --mem=10G                              # Memory pool for all cores
#SBATCH --mail-type=ALL

module load Anaconda3/5.0.1-fasrc02
source activate pcgc

cd ../
mkdir -p ./demo/output

#-------------------------------------------------------
# run the HEELS heritability estimation algorithm
#-------------------------------------------------------

python run_HEELS.py \
    --output_fp ./demo/output/test_h2 \
    --sumstats_fp ./data/ukb_332k_simul_pheno.txt \
    --ld_fp ./data/ukb_332k_chr22_LD.npz \
    --ld_snp_fp ./data/ukb_332k_chr22_maf01_geno1.bim \
    --init_values 0.1,0.9 \
    --N 332340 \
    --tol 1e-4 \
    --calc_var \
    --stream-stdout

