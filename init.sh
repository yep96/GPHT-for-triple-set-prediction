#cd DATA
#cat Family/data_ori/* > Family/data_ori/all_ori.txt
#cat FB15K237/data_ori/* > FB15K237/data_ori/all.txt
#cat WN18RR/data_ori/* > WN18RR/data_ori/all.txt
#python FamilyComplete.py
#python split.py -dataset FB15K237
#python split.py -dataset WN18RR
#python split.py -dataset Family
#cd ..
mkdir -p GPHT/EXPS/FB15K237
mkdir GPHT/EXPS/Family
mkdir GPHT/EXPS/WN18RR
mkdir -p HAKE/EXPS/FB15K237
mkdir HAKE/EXPS/Family
mkdir HAKE/EXPS/WN18RR
mkdir -p RuleTensor-TSP/EXPS/FB15K237
mkdir RuleTensor-TSP/EXPS/Family
mkdir RuleTensor-TSP/EXPS/WN18RR
