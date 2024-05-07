import numpy as np

realDrug_Profile = '../mat_data/mat_drug_protein.txt'
drug_profile="../mat_data/mat_drug_disease.txt"
pro_drugfile="../mat_data/mat_protein_disease.txt"
realDP = np.genfromtxt(realDrug_Profile, delimiter=' ')
DP=np.genfromtxt(drug_profile,delimiter=' ')
PD=np.genfromtxt(pro_drugfile,delimiter=' ')
pdT=PD.T
DTbyDis=np.dot(DP,pdT).astype(int)
def real_fake(realDP,computeDP):
    num = {}
    processed = np.zeros((computeDP.shape[0], computeDP.shape[1]))
    for i in range(computeDP.shape[0]):
        hang = computeDP[i]
        num[i] = []
        for j in range(computeDP.shape[1]):
            if (int(realDP[i][j]) == 1):
                num[i].append(hang[j])
        if (len(num[i]) != 0):
            tmp = np.where(hang >= min(num[i]), 1, 0)
            processed[i] = tmp
    return processed
byDrugM=real_fake(realDP,DTbyDis)
byProM=real_fake(realDP.T,DTbyDis.T).T

pressed = np.zeros((byDrugM.shape[0], byDrugM.shape[1]))
for i in range(byProM.shape[0]):
    pressed[i] = np.int32(np.logical_or(byDrugM[i], byProM[i]))
reafile="mat_drug_protein_negative_sample.txt"
np.savetxt(reafile,pressed,delimiter=' ',fmt="%d")