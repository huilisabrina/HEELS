#!/bin/bash
#SBATCH -n 1                                    # Number of cores
#SBATCH -c 4
#SBATCH -N 1                                    # Ensure that all cores are on one machine
#SBATCH -t 0-1:00                             # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p xlin,xlin-lab,serial_requeue,shared    # Partition to submit to
#SBATCH -o ssld_exact_ld_%A_%a.out
#SBATCH -e ssld_exact_ld_%A_%a.err
#SBATCH --mem=10G                              # Memory pool for all cores
#SBATCH --mail-type=ALL

#-------------------------------------------------------
# customize setup
#-------------------------------------------------------

simul_folder=$1
pheno_tag=$2
causal_type=$3
sample=$4
N=$5
constrain=$6

#-------------------------------------------------------
# load modules
#-------------------------------------------------------
module load Anaconda3/5.0.1-fasrc02
source activate pcgc

#-------------------------------------------------------
# set paths
#-------------------------------------------------------
output_dir="/n/holyscratch01/xlin/huili/HEELS"
code_dir="/n/holystore01/LABS/xlin/Lab/huili/SSLD"
sample_var_fp="${output_dir}/${simul_folder}_${causal_type}/pheno/${pheno_tag}.var"

mkdir -p ${output_dir}/${simul_folder}_${causal_type}/SSLD

cd /n/holystore01/LABS/xlin/Lab/huili/SSLD

# for pheno_index in {1..50}; do
    #-------------------------------------------------------
    # run SSLD for the full chromosome (exact LD, no approx)
    #-------------------------------------------------------
    if [[ ${constrain} == "constrain" ]]; then
        mkdir -p ${output_dir}/${simul_folder}_${causal_type}/SSLD/constrain
        # for pheno_index in {1..2}; do
            # only run SSLD if there are no estimates existing
            # if [[ -f "${output_dir}/${simul_folder}_${causal_type}/SSLD/std_Y/pheno_${SLURM_ARRAY_TASK_ID}.txt" ]]; then
            #     echo "Output file already exists"
            # else
            python ./main_func/run_SSLD.py \
                --output_fp ${output_dir}/${simul_folder}_${causal_type}/SSLD/constrain/pheno_${SLURM_ARRAY_TASK_ID}_LRT_test \
                --est_fp ${output_dir}/${simul_folder}_${causal_type}/SSLD/constrain/pheno_${SLURM_ARRAY_TASK_ID} \
                --sumstats_fp ${output_dir}/${simul_folder}_${causal_type}/sumstats/${sample}_pheno_${SLURM_ARRAY_TASK_ID}.txt \
                --ld_fp ${output_dir}/${simul_folder}_random/gre_LD/std_LD_corr.npz \
                --bim_fp ${output_dir}/${simul_folder}_random/geno/${sample}_chr22_maf01_geno1 \
                --pheno_index ${SLURM_ARRAY_TASK_ID} \
                --constrain_sigma \
                --not_run_ssld \
                --init_values 0.2,0.8 \
                --N ${N} \
                --N_ref ${N} \
                --tol 1e-4 \
                --calc_var \
                --use_lrt \
                --stream-stdout
            # fi
        # done
    elif [[ ${constrain} == "approx" ]]; then
        mkdir -p ${output_dir}/${simul_folder}_${causal_type}/SSLD/std_Y/approx
        # for pheno_index in {1..2}; do
            # only run SSLD if there are no estimates existing
            # if [[ -f "${output_dir}/${simul_folder}_${causal_type}/SSLD/std_Y/pheno_${SLURM_ARRAY_TASK_ID}.txt" ]]; then
            #     echo "Output file already exists"
            # else
            python ./main_func/run_SSLD.py \
                --output_fp ${output_dir}/${simul_folder}_${causal_type}/SSLD/std_Y/approx/pheno_${SLURM_ARRAY_TASK_ID} \
                --sumstats_fp ${output_dir}/${simul_folder}_${causal_type}/sumstats/std_Y/${sample}_pheno_${SLURM_ARRAY_TASK_ID}.txt \
                --ld_fp ${output_dir}/${simul_folder}_random/gre_LD/std_LD_corr.npz \
                --bim_fp ${output_dir}/${simul_folder}_random/geno/${sample}_chr22_maf01_geno1 \
                --pheno_index ${SLURM_ARRAY_TASK_ID} \
                --approx_YtY_bs \
                --init_values 0.2,0.8 \
                --N ${N} \
                --N_ref ${N} \
                --tol 1e-4 \
                --calc_var \
                --stream-stdout
            # fi
        # done
    else
        mkdir -p ${output_dir}/${simul_folder}_${causal_type}/SSLD/std_Y/no_constrain
        # for pheno_index in {1..2}; do
            # only run SSLD if there are no estimates existing
            # if [[ -f "${output_dir}/${simul_folder}_${causal_type}/SSLD/std_Y/pheno_${SLURM_ARRAY_TASK_ID}.txt" ]]; then
            #     echo "Output file already exists"
            # else
            echo "Phenotype ${SLURM_ARRAY_TASK_ID}"
            sample_var=$(awk -v "pheno=${SLURM_ARRAY_TASK_ID}" 'NR == pheno {print $1}' ${sample_var_fp})
            echo "Using sample variance of ${sample_var}"

            python ./main_func/run_SSLD.py \
                --output_fp ${output_dir}/${simul_folder}_${causal_type}/SSLD/std_Y/no_constrain/pheno_${SLURM_ARRAY_TASK_ID} \
                --sumstats_fp ${output_dir}/${simul_folder}_${causal_type}/sumstats/std_Y/${sample}_pheno_${SLURM_ARRAY_TASK_ID}.txt \
                --ld_fp ${output_dir}/${simul_folder}_random/gre_LD/std_LD_corr.npz \
                --bim_fp ${output_dir}/${simul_folder}_random/geno/${sample}_chr22_maf01_geno1 \
                --pheno_index ${SLURM_ARRAY_TASK_ID} \
                --YtY ${sample_var} \
                --init_values 0.2,0.8 \
                --N ${N} \
                --N_ref ${N} \
                --tol 1e-4 \
                --calc_var \
                --stream-stdout
            # fi
        # done
    fi
# done
