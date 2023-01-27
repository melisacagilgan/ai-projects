import numpy as np


class HMM:
    def __init__(self, A, B, Pi):
        self.A = A
        self.B = B
        self.Pi = Pi

    def forward_log(self, O: list):
        """
        :param O: is the sequence (an array of) discrete (integer) observations, i.e. [0, 2,1 ,3, 4]
        :return: ln P(O|λ) score for the given observation, ln: natural logarithm
        """

        # Forward Initialization
        N = self.A.shape[0]
        T = len(O)

        # Alpha: N x T matrix
        alpha = np.zeros((N, T))

        # Initialization step (t = 0)
        for i in range(N):
            alpha[i, 0] = self.Pi[i] * self.B[i, O[0]]

        # Induction step (t > 0)
        ln_P = 0
        for t in range(1, T):
            for i in range(N):
                alpha[i, t] = np.sum(
                    alpha[:, t - 1] * self.A[:, i]) * self.B[i, O[t]]
            # Scaling step
            c_t = 1/np.sum(alpha[:, t])
            alpha[:, t] = c_t * alpha[:, t]

            # ln P(O|λ)
            ln_P -= np.log(c_t)

        return ln_P

    def viterbi_log(self, O: list):
        """
        :param O: is an array of discrete (integer) observations, i.e. [0, 2,1 ,3, 4]
        :return: the tuple (Q*, ln P(Q*|O,λ)), Q* is the most probable state sequence for the given O
        """

        # Viterbi Initialization
        N = self.A.shape[0]
        T = len(O)

        # Delta: N x T matrix
        delta = np.zeros((N, T))

        # Psi: N x T matrix
        psi = np.zeros((N, T))

        # Initialization step (t = 0)
        for i in range(N):
            delta[i, 0] = self.Pi[i] * self.B[i, O[0]]
            psi[i, 0] = 0

        # Induction step (t > 0)
        for t in range(1, T):
            for i in range(N):
                delta[i, t] = np.max(
                    delta[:, t - 1] * self.A[:, i]) * self.B[i, O[t]]
                psi[i, t] = np.argmax(delta[:, t - 1] * self.A[:, i])

        # Termination step
        P = np.max(delta[:, T - 1])

        # Backtracking
        Q_star = np.zeros(T)
        Q_star[T - 1] = np.argmax(delta[:, T - 1])

        for t in range(T - 2, -1, -1):
            Q_star[t] = psi[int(Q_star[t + 1]), t + 1]

        # ln P(Q*|O,λ)
        ln_P = np.log(P)

        return (ln_P, list(Q_star.astype(int)))
