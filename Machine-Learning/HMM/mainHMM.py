from HMM import HMM


import numpy as np

A = np.array([[0.4, 0.6],
              [0.7, 0.3]], dtype=np.longdouble)

B = np.array([[0.3, 0.4, 0.3],
              [0.1, 0.2, 0.7]], dtype=np.longdouble)

Pi = np.array([0.6, 0.4], dtype=np.longdouble)

hmm = HMM(A, B, Pi)

O1 = [2, 1, 0]
O2 = [0, 0, 2, 1, 0]
print(hmm.forward_log(O1))
print(hmm.forward_log(O2))
print(hmm.viterbi_log(O1))
print(hmm.viterbi_log(O2))