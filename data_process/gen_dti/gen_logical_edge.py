from tqdm import tqdm
import numpy as np
positivef = "positive_sample.txt"
nagativef = "negative_sample.txt"
positive = np.genfromtxt(positivef, delimiter=' ')
negative = np.genfromtxt(nagativef, delimiter=' ')
arr = np.vstack((positive, negative)).astype(int)
element_indices = {}
for idx, row in tqdm(enumerate(arr)):
    for element in row:
        if element not in element_indices:
            element_indices[element] = []
        element_indices[element].append(idx)
matching_indices = set()
for indices in tqdm(element_indices.values()):
    if len(indices) > 1:
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                matching_indices.add(tuple(sorted((indices[i], indices[j]))))
matching_indices_array = np.array(list(matching_indices))
np.savetxt("all_edge.txt", matching_indices_array, fmt="%d")
