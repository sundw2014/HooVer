"""
  A collection of very generic python utilities.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import

import numpy as np
import gzip
import pickle
import contextlib

def compare_dict(dict_1, dict_2):
    """ Compares two dictionaries. """
    # N.B: Taken from stackoverflow:
    # http://stackoverflow.com/questions/4527942/comparing-two-dictionaries-in-python
    dict_1_keys = set(dict_1.keys())
    dict_2_keys = set(dict_2.keys())
    intersect_keys = dict_1_keys.intersection(dict_2_keys)
    added = dict_1_keys - dict_2_keys
    removed = dict_2_keys - dict_1_keys
    modified = {o: (dict_1[o], dict_2[o]) for o in intersect_keys if dict_1[o] != dict_2[o]}
    same = set(o for o in intersect_keys if dict_1[o] == dict_2[o])
    return added, removed, modified, same


def dicts_are_equal(dict_1, dict_2):
    """ Returns true if dict_1 and dict_2 are equal. """
    added, removed, modified, _ = compare_dict(dict_1, dict_2)
    return len(added) == 0 and len(removed) == 0 and len(modified) == 0


def map_to_cube(pts, bounds):
    """ Maps bounds to [0,1]^d and returns the representation in the cube. """
    a = (pts - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    return a


def map_to_bounds(pts, bounds):
    """ Given a point in [0,1]^d, returns the representation in the original space. """
    # print(bounds[:][1])
    ### example 1, FIXME
    # a = pts * (bounds[:][1] - bounds[:][0]) + bounds[:][0]
    a = pts * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

    return a


def map_cell_to_bounds(cell, bounds):
    """ Given a point in [0,1]^d, returns the representation in the original space. """
    # cell.shape = [d, 2]
    #bounds.shape = [d, 2]
    # import pdb; pdb.set_trace()
    # return cell
    cell = np.array(cell)
    lower_bounds = map_to_bounds(cell[:, 0], bounds)
    upper_bounds = map_to_bounds(cell[:, 1], bounds)

    return np.stack([lower_bounds, upper_bounds]).T


def compute_average_sq_prediction_error(Y1, Y2):
    """ Returns the average prediction error. """
    return np.linalg.norm(np.array(Y1) - np.array(Y2)) ** 2 / len(Y1)


def dist_squared(X1, X2):
    """ If X1 is n1xd and X2 is n2xd, this returns an n1xn2 matrix where the (i,j)th
      entry is the squared distance between X1(i,:) and X2(j,:).
  """
    n1, dim1 = X1.shape
    n2, dim2 = X2.shape
    if dim1 != dim2:
        raise ValueError('Second dimension of X1 and X2 should be equal.')
    dist_sq = (np.outer(np.ones(n1), (X2 ** 2).sum(axis=1))
               + np.outer((X1 ** 2).sum(axis=1), np.ones(n2))
               - 2 * X1.dot(X2.T))
    return dist_sq


def stable_cholesky(M):
    """ Returns L, a 'stable' cholesky decomposition of M. L is lower triangular and
      satisfies L*L' = M.
      Sometimes nominally psd matrices are not psd due to numerical issues. By adding a
      small value to the diagonal we can make it psd. This is what this function does.
      Use this iff you know that K should be psd. We do not check for errors
  """
    # pylint: disable=superfluous-parens
    if M.size == 0:
        return M  # if you pass an empty array then just return it.
    try:
        # First try taking the Cholesky decomposition.
        L = np.linalg.cholesky(M)
    except np.linalg.linalg.LinAlgError:
        # If it doesn't work, then try adding diagonal noise.
        diag_noise_power = -11
        max_M = np.diag(M).max()
        diag_noise = np.diag(M).max() * 1e-11
        chol_decomp_succ = False
        while not chol_decomp_succ:
            try:
                L = np.linalg.cholesky(M + (10 ** diag_noise_power * max_M) * np.eye(M.shape[0]))
                chol_decomp_succ = True
            except np.linalg.linalg.LinAlgError:
                diag_noise_power += 1
        if diag_noise_power >= 5:
            print('**************** Cholesky failed: Added diag noise = %e' % (diag_noise))
    return L


def draw_gaussian_samples(num_samples, mu, K):
    """ Draws num_samples samples from a Gaussian distribution with mean mu and
      covariance K.
  """
    num_pts = len(mu)
    L = stable_cholesky(K)
    U = np.random.normal(size=(num_pts, num_samples))
    V = L.dot(U).T + mu
    return V

def llist_generator(ll):
    current = ll.first
    yield current
    while current.next is not None:
        current = current.next
        yield current

def savepklz(data_to_dump, dump_file_full_name):
    ''' Saves a pickle object and gzip it '''

    with gzip.open(dump_file_full_name, 'wb') as out_file:
        pickle.dump(data_to_dump, out_file)


def loadpklz(dump_file_full_name):
    ''' Loads a gziped pickle object '''

    with gzip.open(dump_file_full_name, 'rb') as in_file:
        dump_data = pickle.load(in_file)

    return dump_data

def evaluate_single_state(run_markov_chain, init_state, budget, mult=10000):
    init_state = np.array(init_state).reshape(-1).tolist()
    true_max_prob = 0
    for i in range(mult):
        reward = run_markov_chain(init_state.copy(), budget)
        true_max_prob = true_max_prob + reward
    true_max_prob = true_max_prob / mult
    return true_max_prob

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
