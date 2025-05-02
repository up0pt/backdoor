#!/bin/bash
#SBATCH --output=/home/members/nakadam/backdoor/jobs/job%j.out  # where to store the output (%j is the JOBID)
#SBATCH --error=/home/members/nakadam/backdoor/jobs/job%j.err  # where to store error messages
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#CommentSBATCH --nodelist=heart  # you can specify nodes where the job should be run; Ditto

clients=(30)
attackers=(16 8 4)
selections=(random)
rounds=(40)
pdr=(0.7 0.5 0.3)
boost=(20 10 5)

PROJECT_ROOT=/home/members/nakadam/backdoor

echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

cd $PROJECT_ROOT
pwd
nvidia-smi  # show GPU status

for c in "${clients[@]}"; do
    for a in "${attackers[@]}"; do
        for s in "${selections[@]}"; do
            for r in "${rounds[@]}"; do
                for p in "${pdr[@]}"; do
                    for b in "${boost[@]}"; do
                        echo ">>> clients=$c, attackers=$a, selection=$s, rounds=$r, pdr=$p, boost=$b"
                        uv run src/main.py \
                        --clients  $c \
                        --num_attackers  $a \
                        --attack_selection  $s \
                        --rounds  $r \
                        --pdr     $p \
                        --boost   $b \
                        --topology   barabasi \
                        --seed    123 \
                        --m       3
                    done
                done
            done
        done
    done
done

# Send more noteworthy information to the output log
echo "Finished at: $(date)"

# End the script with exit code 0
exit 0
