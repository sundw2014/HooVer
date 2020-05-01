# Author: Rajat Sen

from __future__ import print_function
from __future__ import division

import os
import sys
import random
import sys
import time
import numpy as np
import math
import datetime
#from mf.mf_func import MFOptFunction  # MF function object
from utils.general_utils import map_to_cube  # mapping everything to [0,1]^d cube
# from examples.synthetic_functions import *
from multiprocessing import Process
# import brewer2mpl

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# -----------------------------------------------------------------------------

nu_mult = 1  # multiplier to the nu parameter

# -----------------------------------------------------------------------------


def flip(p):
    return True if random.random() < p else False

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class MF_node(object):

    def __init__(self, cell, value, fidel, upp_bound, height, dimension, num):
        """This is a node of the MFTREE
        cell: tuple denoting the bounding boxes of the partition
        m_value: mean value of the observations in the cell and its children
        value: value in the cell
        fidelity: the last fidelity that the cell was queried with
        upp_bound: B_{i,t} in the paper
        t_bound: upper bound with the t dependent term
        height: height of the cell (sometimes can be referred to as depth in the tree)
        dimension: the dimension of the parent that was halved in order to obtain this cell
        num: number of queries inside this partition so far
        left,right,parent: pointers to left, right and parent
        """
        self.cell = cell
        self.m_value = value
        self.value = value
        self.fidelity = fidel
        self.upp_bound = upp_bound
        self.height = height
        self.dimension = dimension
        self.num = num
        self.t_bound = upp_bound

        self.left = None
        self.right = None
        self.parent = None

    # -----------------------------------------------------------------------------

    def __cmp__(self, other):
        return cmp(other.t_bound, self.t_bound)

    # -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def in_cell(node, parent):
    """Check if 'node' is a subset of 'parent' node can
       either be a MF_node or just a tuple denoting its cell
    """
    try:
        ncell = list(node.cell)
    except:
        ncell = list(node)

    pcell = list(parent.cell)
    flag = 0

    for i in range(len(ncell)):
        if ncell[i][0] >= pcell[i][0] and ncell[i][1] <= pcell[i][1]:
            flag = 0
        else:
            flag = 1
            break
    if flag == 0:
        return True
    else:
        return False

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class MF_tree(object):
    """
    MF_tree class that maintains the multi-fidelity tree
    nu: nu parameter in the paper
    rho: rho parameter in the paper
    sigma: noise variance, ususally a hyperparameter for the whole process
    C: parameter for the bias function as defined in the paper
    root: can initialize a root node, when this parameter is supplied by a MF_node object instance
    """

    # -----------------------------------------------------------------------------

    def __init__(self, nu, rho, sigma, C, root=None):
        self.nu = nu
        self.rho = rho
        self.sigma = sigma
        self.root = root
        self.C = C
        self.root = root
        self.mheight = 0
        self.maxi = float(-sys.maxsize - 1)
        self.current_best = root
        self.current_best_cell = {}

    # -----------------------------------------------------------------------------

    def insert_node(self, root, node):
        """ insert a node in the tree in the appropriate position """
        if self.root is None:
            node.height = 0
            if self.mheight < node.height:
                self.mheight = node.height
            self.root = node
            self.root.parent = None
            return self.root
        if root is None:
            node.height = 0
            if self.mheight < node.height:
                self.mheight = node.height
            root = node
            root.parent = None
            return root
        if root.left is None and root.right is None:
            node.height = root.height + 1
            if self.mheight < node.height:
                self.mheight = node.height
            root.left = node
            root.left.parent = root
            return root.left
        elif root.left is not None:
            if in_cell(node, root.left):
                return self.insert_node(root.left, node)
            elif root.right is not None:
                if in_cell(node, root.right):
                    return self.insert_node(root.right, node)
            else:
                node.height = root.height + 1
                if self.mheight < node.height:
                    self.mheight = node.height
                root.right = node
                root.right.parent = root
                return root.right

    # -----------------------------------------------------------------------------

    def update_parents(self, node, val):
        """
        update the upperbound and mean value of a parent node, once a new child is inserted in its child tree. This process proceeds recursively up the tree
        """
        if node.parent is None:
            return
        else:
            parent = node.parent
            parent.m_value = (parent.num * parent.m_value + val) / (1.0 + parent.num)
            parent.num = parent.num + 1.0
            parent.upp_bound = parent.m_value + 2 * ((self.rho) ** (parent.height)) * self.nu
            self.update_parents(parent, val)

    # -----------------------------------------------------------------------------

    def update_tbounds(self, root, t):
        """ updating the tbounds of every node recursively """
        if root is None:
            return
        self.update_tbounds(root.left, t)
        self.update_tbounds(root.right, t)
        root.t_bound = root.upp_bound + np.sqrt(2 * (self.sigma ** 2) * np.log(t) / root.num)
        maxi = None
        if root.left:
            maxi = root.left.t_bound
        if root.right:
            if maxi:
                if maxi < root.right.t_bound:
                    maxi = root.right.t_bound
            else:
                maxi = root.right.t_bound
        if maxi:
            root.t_bound = min(root.t_bound, maxi)

    # -----------------------------------------------------------------------------

    def print_given_height(self, root, height):
        if root is None:
            return
        if root.height == height:
            print(root.cell, root.num, root.upp_bound, root.t_bound),
        elif root.height < height:
            if root.left:
                self.print_given_height(root.left, height)
            if root.right:
                self.print_given_height(root.right, height)
        else:
            return

    # -----------------------------------------------------------------------------

    def levelorder_print(self):
        """ levelorder print """
        for i in range(self.mheight + 1):
            self.print_given_height(self.root, i)
            print('\n')

    # -----------------------------------------------------------------------------

    def search_cell(self, root, cell):
        """ check if a cell is present in the tree """
        if root is None:
            return False, None, None
        if root.left is None and root.right is None:
            if root.cell == cell:
                return True, root, root.parent
            else:
                return False, None, root
        if root.left:
            if in_cell(cell, root.left):
                return self.search_cell(root.left, cell)
        if root.right:
            if in_cell(cell, root.right):
                return self.search_cell(root.right, cell)

    # -----------------------------------------------------------------------------

    def get_next_node(self, root):
        """
        getting the next node to be queried or broken, see the algorithm in the paper
        """
        if root is None:
            print('Could not find next node. Check Tree.')
        if root.left is None and root.right is None:
            return root
        if root.left is None:
            return self.get_next_node(root.right)
        if root.right is None:
            return self.get_next_node(root.left)

        if root.left.t_bound > root.right.t_bound:
            return self.get_next_node(root.left)
        elif root.left.t_bound < root.right.t_bound:
            return self.get_next_node(root.right)
        else:
            bit = flip(0.5)
            if bit:
                return self.get_next_node(root.left)
            else:
                return self.get_next_node(root.right)

    # -----------------------------------------------------------------------------

    def get_current_best(self, root):
        """
        get current best cell from the tree
        """
        if root is None:
            return
        if root.right is None and root.left is None:
            val = root.m_value - self.nu * ((self.rho) ** (root.height))
            if self.maxi < val:
                self.maxi = val
                cell = list(root.cell)
                self.current_best_cell = root.parent.cell
                self.current_best = np.array([(s[0] + s[1]) / 2.0 for s in cell])

            return
        if root.left:
            self.get_current_best(root.left)
        if root.right:
            self.get_current_best(root.right)
    # -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class MFHOO(object):
    """
    MFHOO algorithm, given a fixed nu and rho
    mfobject: multi-fidelity noisy function object
    nu: nu parameter
    rho: rho parameter
    budget: total budget provided either in units or time in seconds
    sigma: noise parameter
    C: bias function parameter
    tol: default parameter to decide whether a new fidelity query is required for a cell
    Randomize: True implies that the leaf is split on a randomly chosen dimension, False means the scheme in DIRECT algorithm is used. We recommend using False.
    Auto: Select C automatically, which is recommended for real data experiments
    CAPITAL: 'Time' mean time in seconds is used as cost unit, while 'Actual' means unit cost used in synthetic experiments
    debug: If true then more messages are printed
    """

    # -----------------------------------------------------------------------------

    def __init__(self, mfobject, nu, rho, budget, sigma, C, tol=1e-3, \
                 Randomize=False, Auto=True, value_dict={}, \
                 CAPITAL='Time', debug='True', comp_value_dict = {}, useHOO=False):
        self.num_query = 0
        self.mfobject = mfobject
        self.nu = nu
        self.rho = rho
        self.budget = int(budget)
        self.C = 1 * C
        self.t = 0
        self.sigma = sigma
        self.tol = tol
        self.Randomize = Randomize
        self.cost = 0
        self.cflag = False
        self.value_dict = value_dict
        self.comp_value_dict = comp_value_dict
        self.CAPITAL = CAPITAL
        self.debug = debug
        self.useHOO = useHOO
        if Auto:
            z1 = 0.8
            z2 = 0.2
            d = self.mfobject.domain_dim
            x = np.array([0.5] * d)
            t1 = time.time()
            # max_iteration = 50
            if not self.useHOO:
                v1 = self.mfobject.eval_at_fidel_single_point_normalised_average(z1, x, mfobject.max_iteration)
                v2 = self.mfobject.eval_at_fidel_single_point_normalised_average(z2, x, mfobject.max_iteration)
            else:
                v1 = self.mfobject.eval_at_fidel_single_point_normalised_average(z1, x, mfobject.max_iteration) # FIXME: old code use eval_at_fidel_single_point_normalised which only run simulation for one time, here we run mfobject.max_iteration to make it consistent with MFHOO. However, HOO actually doesn't reach here, it doesn't use Auto
                v2 = self.mfobject.eval_at_fidel_single_point_normalised_average(z2, x, mfobject.max_iteration)

            t2 = time.time()
            self.C = 1 * max(np.sqrt(2) * np.abs(v1 - v2) / np.abs(z1 - z2), 0.01)
            # self.C = np.sqrt(2) * np.abs(v1 - v2) / np.abs(z1 - z2)
            # if self.C == 0:
            #     self.Auto = False
            #     self.nu = self.nu
            # else:
            #     self.Auto = True
            #     self.nu = nu_mult * self.C

            self.nu = nu_mult * self.C
            if self.debug:
                print('Auto Init: ')
                print('C: ' + str(self.C))
                print('nu: ' + str(self.nu))
            c1 = self.mfobject.eval_fidel_cost_single_point_normalised(z1)
            c2 = self.mfobject.eval_fidel_cost_single_point_normalised(z2)
            self.cost = c1 + c2
            if self.CAPITAL == 'Time':
                self.cost = t2 - t1
        d = self.mfobject.domain_dim
        cell = tuple([(0, 1)] * d)
        height = 0
        if not self.useHOO:
            dimension = 1
            root, cost = self.querie(cell, height, self.rho, self.nu, dimension, option=1)
        else:
            dimension = 0
            root, cost = self.querie(cell, height, self.rho, self.nu, dimension, option=0)

        self.t = self.t + 1
        self.Tree = MF_tree(nu, rho, self.sigma, C, root)
        self.Tree.update_tbounds(self.Tree.root, self.t)
        self.cost = self.cost + cost

    # -----------------------------------------------------------------------------

    def get_value(self, cell, fidel):
        """cell: tuple"""
        x = np.array([(s[0] + s[1]) / 2.0 for s in list(cell)])
        return self.mfobject.eval_at_fidel_single_point_normalised(fidel, x)

    # -----------------------------------------------------------------------------

    def querie(self, cell, height, rho, nu, dimension, option=1):
        diam = nu * (rho ** height)
        if option == 1:
            # if self.C == 0.0:
            #     z = 1
            # else:
            #     z = min(max(1 - diam / self.C, self.tol), 1.0)
            z = min(max(1 - diam / self.C, self.tol), 1.0)
        else:
            z = 1.0
        if False:#cell in self.value_dict: # disable cache for fair comparison. FIXME: a better idea is to change L418-420, check z also when check the cache, add new cell if z is different.
            current = self.value_dict[cell]
            if abs(current.fidelity - z) <= self.tol:
                value = current.value
                cost = 0
            else:
                t1 = time.time()
                value = self.get_value(cell, z)
                t2 = time.time()
                if abs(value - current.value) > self.C * abs(current.fidelity - z):
                    self.cflag = True
                current.value = value
                current.m_value = value
                current.fidelity = z
                self.value_dict[cell] = current
                if self.CAPITAL == 'Time':
                    cost = t2 - t1
                else:
                    cost = self.mfobject.eval_fidel_cost_single_point_normalised(z)
        else:
            t1 = time.time()
            value = self.get_value(cell, z)
            t2 = time.time()
            bhi = 2 * diam + value
            self.value_dict[cell] = MF_node(cell, value, z, bhi, height, dimension, 1)
            if self.CAPITAL == 'Time':
                cost = t2 - t1
            else:
                cost = self.mfobject.eval_fidel_cost_single_point_normalised(z)

        bhi = 2 * diam + value
        current_object = MF_node(cell, value, z, bhi, height, dimension, 1)
        return current_object, cost

    # -----------------------------------------------------------------------------

    def update_comp_value_dict(self, node):
        """
        update the upperbound and mean value of a parent node, once a new child is inserted in its child tree. This process proceeds recursively up the tree
        """
        if node.parent is None:
            return
        else:
            parent = node.parent

            self.comp_value_dict[parent.cell] = parent
            self.update_comp_value_dict(parent)

    # -----------------------------------------------------------------------------

    def split_children(self, current, rho, nu, option=1):
        pcell = list(current.cell)
        span = [abs(pcell[i][1] - pcell[i][0]) for i in range(len(pcell))]
        if self.Randomize:
            dimension = np.random.choice(range(len(pcell)))
        else:
            dimension = np.argmax(span)
        # dimension = 1  # partition the environment along y direction
        dd = len(pcell)
        if dimension == current.dimension:
            dimension = (current.dimension - 1) % dd
        # dimension = 1  # partition the environment along y direction
        cost = 0
        h = current.height + 1
        l = np.linspace(pcell[dimension][0], pcell[dimension][1], 3)
        children = []
        for i in range(len(l) - 1):
            cell = []
            for j in range(len(pcell)):
                if j != dimension:
                    cell = cell + [pcell[j]]
                else:
                    cell = cell + [(l[i], l[i + 1])]
            cell = tuple(cell)
            # _t1 = time.time()
            child, c = self.querie(cell, h, rho, nu, dimension, option)
            # _t2 = time.time()
            # real_cost = _t2 - _t1
            # import pdb; pdb.set_trace()
            children = children + [child]
            cost = cost + c

        return children, cost

    # -----------------------------------------------------------------------------

    def take_HOO_step(self):
        # _t0 = time.time()
        current = self.Tree.get_next_node(self.Tree.root)
        # _t1 = time.time()
        children, cost = self.split_children(current, self.rho, self.nu, 1 if not self.useHOO else 0)
        # _t2 = time.time()
        # import pdb; pdb.set_trace()
        self.t = self.t + 2
        self.cost = self.cost + cost
        rnode = self.Tree.insert_node(self.Tree.root, children[0])
        self.Tree.update_parents(rnode, rnode.value)
        self.update_comp_value_dict(rnode)
        rnode = self.Tree.insert_node(self.Tree.root, children[1])
        self.Tree.update_parents(rnode, rnode.value)
        self.update_comp_value_dict(rnode)
        self.Tree.update_tbounds(self.Tree.root, self.t)
        # _t3 = time.time()
        # import pdb; pdb.set_trace()
    # -----------------------------------------------------------------------------

    def run(self):
        self.num_query = 0
        #import pdb; pdb.set_trace()
        while self.t+2 <= self.budget:
            # old_cost = self.cost
            # _t1 = time.time()
            self.num_query = self.num_query + 2
            self.take_HOO_step()
            # _t2=time.time()
            # print('%.5f, %.5f, %.1f'%(self.cost-old_cost, _t2-_t1, (_t2-_t1)/(self.cost-old_cost)))
        #print('number of queries: ' + str(self.num_query))
        print('number of queries: ' + str(self.t))

    # -----------------------------------------------------------------------------

    def get_point(self):
        self.Tree.get_current_best(self.Tree.root)
        return self.Tree.current_best, self.Tree.current_best_cell
    # -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class MFPOO(object):
    """
    MFPOO object that spawns multiple MFHOO instances
    """

    # -----------------------------------------------------------------------------

    def __init__(self, mfobject, nu_max, rho_max, nHOO, sigma, C, mult, tol=1e-3, Randomize=False, Auto=True,
                 unit_cost=1.0, CAPITAL='Time', debug='True', useHOO=False, direct_budget=0.2):
        self.number_of_queries = []
        self.mfobject = mfobject
        self.nu_max = nu_max
        self.rho_max = rho_max
        self.nHOO = nHOO
        self.budget = direct_budget
        self.C = 1* C
        self.t = 0
        self.sigma = sigma
        self.tol = tol
        self.Randomize = Randomize
        self.cost = 0
        self.value_dict = {}
        self.comp_value_dict = {}
        self.MH_arr = []
        self.CAPITAL = CAPITAL
        self.debug = debug
        self.useHOO = useHOO
        if useHOO: assert Auto==False

        if Auto:
            if unit_cost is None:
                z1 = 1.0
                if self.debug:
                    print('Setting unit cost automatically as None was supplied')
            else:
                z1 = 0.8
            z2 = 0.2
            d = self.mfobject.domain_dim
            x = np.array([0.5] * d)
            t1 = time.time()
            # max_iteration = 50
            if not self.useHOO:
                v1 = self.mfobject.eval_at_fidel_single_point_normalised_average(z1, x, mfobject.max_iteration)
            else:
                v1 = self.mfobject.eval_at_fidel_single_point_normalised([z1], x)
            t3 = time.time()
            if not self.useHOO:
                v2 = self.mfobject.eval_at_fidel_single_point_normalised_average(z2, x, mfobject.max_iteration)
            else:
                v2 = self.mfobject.eval_at_fidel_single_point_normalised([z2], x)
            t2 = time.time()
            self.C = 1 * max(np.sqrt(2) * np.abs(v1 - v2) / np.abs(z1 - z2), 0.01)
            # self.C = np.sqrt(2) * np.abs(v1 - v2) / np.abs(z1 - z2)
            # if self.C == 0:
            #     self.Auto = False
            #     self.nu_max = self.nu_max
            # else:
            #     self.Auto = True
            #     self.nu_max = nu_mult * self.C
            self.nu_max = nu_mult * self.C
            if unit_cost is None:
                unit_cost = t3 - t1
                if self.debug:
                    print('Unit Cost: ', unit_cost)
            if self.debug:
                print('Auto Init: ')
                print('C: ' + str(self.C))
                print('nu: ' + str(self.nu_max))
            c1 = self.mfobject.eval_fidel_cost_single_point_normalised(z1)
            c2 = self.mfobject.eval_fidel_cost_single_point_normalised(z2)

        # if self.CAPITAL == 'Time':
        #     self.unit_cost = unit_cost
        # else:
        #     self.unit_cost = self.mfobject.eval_fidel_cost_single_point_normalised(1.0)
        if self.debug:
            print('Number of MFHOO Instances: ' + str(self.nHOO))
            print('Budget per MFHOO Instance:' + str(self.budget))

    # -----------------------------------------------------------------------------

    def run_all_MFHOO(self):
        nu = self.nu_max
        self.number_of_queries = []
        for i in range(self.nHOO):
            rho = self.rho_max ** (float(self.nHOO) / (self.nHOO - i))
            MH = MFHOO(mfobject=self.mfobject, nu=nu, rho=rho, budget=self.budget, sigma=self.sigma, C=self.C, tol=1e-3,
                       Randomize=False, Auto=True if not self.useHOO else False, value_dict=self.value_dict, CAPITAL=self.CAPITAL, debug=self.debug, comp_value_dict = self. comp_value_dict, useHOO=self.useHOO)
            print('Running SOO number: ' + str(i + 1) + ' rho: ' + str(rho) + ' nu: ' + str(nu))
            MH.run()

            _a, _b = MH.get_point()
            _node = MH.comp_value_dict[_b]
            _ncell = self.mfobject.get_unnormalised_cell(_node.cell)

            print('depth: %d, ncell:'%(_node.height)+str(_ncell))
            # node_fidelity[i] = node.fidelity
            # mean_values[i] = node.m_value
            i = i + 1

            #self.number_of_queries = self.number_of_queries + [MH.num_query]
            self.number_of_queries = self.number_of_queries + [MH.t]
            print('Done!')
            self.cost = self.cost + MH.cost
            if MH.cflag:
                self.C = 1.4 * self.C
                # if self.C == 0:
                #     nu = self.nu_max
                #     MH.Auto = False
                # else:
                #     nu = nu_mult * self.C
                #     self.nu_max = nu_mult * self.C
                #     MH.Auto = True
                nu = nu_mult * self.C
                self.nu_max = nu_mult * self.C
                if self.debug:
                    print('Updating C')
                    print('C: ' + str(self.C))
                    print('nu_max: ' + str(nu))
            self.value_dict = MH.value_dict
            self. comp_value_dict = MH. comp_value_dict
            self.MH_arr = self.MH_arr + [MH]

    # -----------------------------------------------------------------------------

    def get_point(self):
        points = np.zeros((len(self.MH_arr), self.mfobject.domain_dim))
        # node_fidelity = np.zeros(len(self.MH_arr))
        # mean_values = np.zeros(len(self.MH_arr))
        node_depth = np.zeros(len(self.MH_arr))
        d = self.mfobject.domain_dim
        Cells = []
        i = 0
        for H in self.MH_arr:
            a, b = H.get_point()
            points[i][:] = a
            node = H.comp_value_dict[b]
            ncell = self.mfobject.get_unnormalised_cell(node.cell)
            Cells = Cells + [ncell]
            node_depth[i] = node.height
            # node_fidelity[i] = node.fidelity
            # mean_values[i] = node.m_value
            i = i + 1

        newp = []
        for p in points:
            _, npoint = self.mfobject.get_unnormalised_coords(None, p)
            newp = newp + [npoint]

        # node_fidelity_unnormalized = []
        # for f in node_fidelity:
        #     fidelity_unnormalized, _ = self.mfobject.get_unnormalised_coords(f, None)
        #     node_fidelity_unnormalized = node_fidelity_unnormalized + [fidelity_unnormalized]

        # return newp, node_fidelity_unnormalized
        return newp, node_depth, Cells
        # return newp
