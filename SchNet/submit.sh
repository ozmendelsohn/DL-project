#BSUB -J spbbr3-SchNet
#BSUB -q new-all.q
#BSUB -R rusage[mem=10GB] affinity[core*1]
#BSUB -u ozyosef.mendelsohn@weizmann.ac.il
python -u auto-sub.py --queue=gpu-medium >> logy.log