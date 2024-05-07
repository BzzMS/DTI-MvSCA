import numpy as np

matPf = '../mat_data/mat_drug_protein.txt'
matP = np.genfromtxt(matPf, delimiter=' ')
P_row, P_col = np.where(matP == 1)
choose_positive=[]
for i in range(len(P_row)):
    x=P_row[i]
    y=P_col[i]
    choose_positive.append([x,y])
aac=np.array(choose_positive)
pos_label = np.ones((len(P_row), 1)).astype(int)
reafile="positive_sample.txt"
np.savetxt(reafile,aac,fmt="%d")

#negative
matP = np.genfromtxt(matPf, delimiter=' ')
matNf = '../negative_sampler/mat_drug_protein_negative_sample.txt'
matN = np.genfromtxt(matNf, delimiter=' ')
countP = np.count_nonzero(matP)
N_row, N_col = np.where(matN == 0)
index = np.random.choice(range(len(N_row)), replace=False, size=countP)
choose_negative=[]
for i in index:
    x=N_row[i]
    y=N_col[i]
    choose_negative.append([x,y])
aac=np.array(choose_negative)
neg_label = np.zeros((countP, 1)).astype(int)
label = np.vstack((pos_label,neg_label))
np.savetxt("label.txt",label,fmt="%d")
reafile="negative_sample.txt"
np.savetxt(reafile,aac,fmt="%d")