import numpy as np
import pandas as pd
import time
from numpy.linalg import norm
import math
import random
from scripts.plots import plot_embeddings
from tqdm import tqdm


class Poincare_Embeddings(object):

    # Initialize class with number of epoch, number of negative sampling, learning rate and threshold for projection on
    # the Poincaré ball
    def __init__(self, dataset, keep=200, nb_epoch=10, nb_neg_sampling=10, lr=0.1, eps=1e-6):
        self.dataset = dataset
        self.dim = 2
        self.nb_epoch = nb_epoch
        self.nb_neg_sampling = nb_neg_sampling
        self.learning_rate = lr
        self.eps = eps
        self.keep = keep

        all_data = pd.read_csv('data/%s_network.csv' % dataset, sep=',', header=None)
        initial_nb_vertex = len(all_data)

        #Keep random part of dataset
        if len(all_data) < keep:
            kept_nb_vertexes = len(all_data)
            pass
        else:
            kept_nb_vertexes = self.keep
            all_data = all_data.sample(n=self.keep).reset_index(drop=True)

        print('%s links kept out of %s' % (kept_nb_vertexes, initial_nb_vertex))
        styles = list(set(all_data[0].append(all_data[1])))

        self.links = [[all_data[0][i], all_data[1][i]] for i in range(len(all_data))]
        self.words = {styles[i]: i for i in range(len(styles))}
        self.embeddings = [np.random.uniform(low=-0.001, high=0.001, size=(self.dim,)) for i in range(len(self.words))]

    def projected_gradient_descent(self, old, gradient, lr):
        """
        Computes the gradient descent and the projection on Poincaré ball

        Arguments:
        old -- previous value
        gradient -- gradient value
        lr -- learning rate

        Returns:
        Updated value by RSGD and projected on Poincaré ball
        """
        eps = self.eps
        new = old - lr * gradient
        if norm(new) >= 1:
            return new / norm(new) - eps
        else:
            return new

    def dist_and_metrics(self, u, v):

        """
        Computes distance and various useful metrics regarding two vectors u and v

        Arguments:
        u -- first embedded object
        v -- second embedded object

        Returns:
         -- the distance between the two objects
        u -- first embedded object
        v -- second embedded object
        norm_u -- squared norm of u
        norm_v -- squared norm of v
        u_scalar_v -- scalar product between u and v
        alpha -- constant alpha as defined in the paper
        beta -- constant beta as defined in the paper
        gamma -- constant gamma as defined in the paper
        """

        norm_u = norm(u) ** 2
        norm_v = norm(v) ** 2
        u_scalar_v = (u * v).sum()

        alpha = 1 - norm_u
        beta = 1 - norm_v

        #Calculate gamma and replace value by 1 if greater to be able to take arccosh
        gamma = max(1., 1 + 2 * (norm_u + norm_v - 2 * u_scalar_v) / (alpha * beta))

        return np.arccosh(gamma), (u, v, norm_u, u_scalar_v, norm_v, alpha, beta, gamma)

    def partial(self, left_partial, metrics):
        """
        Computes complete partial for gradient descent given the left_partial and the metrics (see paper and article for
        the justification of the formula)

        Arguments:
        left_partial -- left side of the gradient (see article and report)
        metrics -- metrics

        Returns:
        Complete gradient
        """
        left_partial = left_partial
        u, v, norm_u, u_scalar_v, norm_v, alpha, beta, gamma = metrics
        if gamma == 1:
            return None, None
        constant = left_partial * 1. / (((gamma ** 2 - 1) ** 0.5) * alpha * beta)
        constant_u = constant * alpha ** 2
        constant_v = constant * beta ** 2
        gradient_u = constant_u * (u * (1 + norm_v - 2 * u_scalar_v) / alpha - v)
        gradient_v = constant_v * (v * (1 + norm_u - 2 * u_scalar_v) / beta - u)

        return gradient_u, gradient_v

    def negative_sampling(self, u, v, words, embeddings):
        """
        Computes the negative sampling for two words of the dictionnary

        Arguments:
        u -- first word
        v -- second word

        Returns:
        list of negative samplings:
        """
        length = len(words)
        i1 = words[u]
        i2 = words[v]
        distance, metrics = self.dist_and_metrics(embeddings[i1], embeddings[i2])
        yield i1, i2, math.exp(-distance), metrics
        for _ in range(self.nb_neg_sampling):
            s1, s2 = random.choice(range(length)), random.choice(range(length))
            distance, metrics = self.dist_and_metrics(embeddings[s1], embeddings[s2])
            yield s1, s2, math.exp(-distance), metrics

    def train_model(self):
        links, words, embeddings = self.links, self.words, self.embeddings
        print('Running Poincaré Embeddings for dataset %s...' % self.dataset)
        time.sleep(0.5)
        for epoch in tqdm(range(self.nb_epoch)):
            random.shuffle(links)

            # Change rate with time according to article
            if epoch < 11:
                eta = self.learning_rate / 10
            else:
                eta = self.learning_rate

            # Go through words of dataset
            for w1, w2 in links:

                # Calculate the denominator of the loss
                exp_neg_dists = list(self.negative_sampling(w1, w2, words, embeddings))
                denom = sum(l[2] for l in exp_neg_dists)

                # Apply gradient descent for each negative sampling iteration
                for i, (i1, i2, distance, metrics) in enumerate(exp_neg_dists):

                    left_partial = 1. - distance / denom if i == 0 else - distance / denom

                    gradient_w1, gradient_w2 = self.partial(left_partial, metrics)

                    if gradient_w1 is not None:
                        embeddings[i1] = self.projected_gradient_descent(embeddings[i1], gradient_w1, eta)
                    if gradient_w2 is not None:
                        embeddings[i2] = self.projected_gradient_descent(embeddings[i2], gradient_w2, eta)

        # print(links, words, embeddings)
        plot_embeddings(self.links, self.words, self.embeddings, self.dataset, self.nb_epoch)