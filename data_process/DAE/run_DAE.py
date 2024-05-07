import numpy as np
from DAE import DAE
import random


def run_dae():
    drug_train = np.loadtxt(r"../feature/drug_vector.txt")
    protein_train = np.loadtxt(r"../feature/protein_vector.txt")

    drug_size=drug_train.shape[1]
    protein_size=protein_train.shape[1]

    print(drug_size,protein_size)
    drug_feature=DAE(drug_train,drug_size,20,16,1,100,[100])
    np.savetxt('../feature/drug_dae_d100.txt',drug_feature)

    protein_feature=DAE(protein_train,protein_size,20,32,1,400,[400])
    np.savetxt('../feature/protein_dae_d400.txt',protein_feature)
    return drug_feature, protein_feature



if __name__ == "__main__":
    drug_feature, protein_feature = run_dae()
    # generate_pair()
