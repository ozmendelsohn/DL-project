#BSUB -J DL-proj
#BSUB -q new-all.q
#BSUB -u ozyosef.mendelsohn@weizmann.ac.il
#BSUB -R rusage[mem=10GB]
#BSUB -R affinity[core*1]
#BSUB -u ozyosef.mendelsohn@weizmann.ac.il
#BSUB -oe DL-proj.e%J
#BSUB -oo DL-proj.o%J
source ~/.bashrc
conda activate sgdml
python -u grid_search_div.py --queue=gpu-short >> logy.log
