import numpy as np
positivef = "positive_sample.txt"
nagativef = "negative_sample.txt"
drugf = '../feature/drug_dae_d100.txt'
prof = '../feature/protein_dae_d400.txt'
positive = np.genfromtxt(positivef, delimiter=' ')
negative = np.genfromtxt(nagativef, delimiter=' ')
drug_gc = np.genfromtxt(drugf, delimiter=' ')
pro_gc = np.genfromtxt(prof, delimiter=' ')
all_dti = np.vstack((positive, negative)).astype(int)
features=[]
for i in all_dti:
    features.append(np.hstack((drug_gc[i[0]],pro_gc[i[1]])))
features=np.array(features)
reafile="features.txt"
np.savetxt(reafile,features)
