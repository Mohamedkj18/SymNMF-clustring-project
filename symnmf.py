import numpy as np
import pandas as pd
import sys
import symnmfsp as s

np.random.seed(1234)


 


def symnmf(goal, k , path, std=False):
    dataPoints = pd.read_csv(path, header=None)
    dataPoints = dataPoints.values.tolist()


    if goal == "symnmf":
        normMat = s.norm(dataPoints)
        m = sum(sum(normMat[i]) for i in range(len(normMat))) / (len(normMat) ** 2)
        H = [[np.random.uniform(0, 2 * ((m / k) ** 0.5)) for i in range(k)] for i in range(len(dataPoints))]
        mat = s.symnmf(normMat, H, k, len(dataPoints))
    elif goal == "sym":

        mat = s.sym(dataPoints)
    elif goal == "ddg":
        mat = s.ddg(dataPoints)
    else:
        mat = s.norm(dataPoints)


    if std:
        for row in mat:
            r = ",".join(f"{num:.4f}" for num in row)
            print(r)
    
    return mat

if __name__ == "__main__":
    k = 0
    p = 1
    if len(sys.argv) == 4:
        k = int(sys.argv[p])
    else:
        p = -1
    goal = sys.argv[p + 1]
    path = sys.argv[p + 2]

    symnmf(goal, k, path, True)

