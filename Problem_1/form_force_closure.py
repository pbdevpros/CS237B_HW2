import cvxpy as cp
import numpy as np

def cross_matrix(x):
    """
    Returns a matrix x_cross such that x_cross.dot(y) represents the cross
    product between x and y.

    For 3D vectors, x_cross is a 3x3 skew-symmetric matrix. For 2D vectors,
    x_cross is a 2x1 vector representing the magnitude of the cross product in
    the z direction.
     """
    D = x.shape[0]
    if D == 2:
        return np.array([[-x[1], x[0]]])
    elif D == 3:
        return np.array([[0., -x[2], x[1]],
                         [x[2], 0., -x[0]],
                         [-x[1], x[0], 0.]])
    raise RuntimeError("cross_matrix(): x must be 2D or 3D. Received a {}D vector.".format(D))

def wrench(f, p):
    """
    Computes the wrench from the given force f applied at the given point p.
    Works for 2D and 3D.

    Args:
        f - 2D or 3D contact force.
        p - 2D or 3D contact point.

    Return:
        w - 3D or 6D contact wrench represented as (force, torque).    
    """
    ########## Your code starts here ##########
    # Hint: you may find cross_matrix(x) defined above helpful. This should be one line of code.
    w = (f, cross_matrix(p).dot(f))
    ########## Your code ends here ##########

    return w

def cone_edges(f, mu):
    """
    Returns the edges of the specified friction cone. For 3D vectors, the
    friction cone is approximated by a pyramid whose vertices are circumscribed
    by the friction cone.

    In the case where the friction coefficient is 0, a list containing only the
    original contact force is returned.

    Args:
        f - 2D or 3D contact force.
        mu - friction coefficient.

    Return:
        edges - a list of forces whose convex hull approximates the friction cone.
    """
    # Edge case for frictionless contact
    if mu == 0.:
        return [f]

    # Planar wrenches
    D = f.shape[0]
    if D == 2:
        ########## Your code starts here ##########
        edges = [np.zeros(D)] * 2
        # fx, fy = f[0], f[1]
        beta = np.arctan(mu * np.pi)
        # fz = np.linalg.norm(f)
        # beta = np.arctan(mu * np.pi)
        # offset =  np.pi / 2
        # edges[0][0] = (fz * np.cos(-beta))
        # edges[0][1] = (fz * np.sin(-beta))
        # edges[1][0] = (fz * np.cos(beta))
        # edges[1][1] = (fz * np.sin(beta))
        R = np.array([
            [np.cos(beta), -np.sin(beta)],
            [np.sin(beta), np.cos(beta)]
        ])
        edges[0] = f.dot(R)
        R = np.array([
            [np.cos(beta), np.sin(beta)],
            [-np.sin(beta), np.cos(beta)]
        ])
        edges[1] = f.dot(R)
        ########## Your code ends here ##########

    # Spatial wrenches
    elif D == 3:
        ########## Your code starts here ##########
        def rotz(th, b):
            return np.array([
                [    np.cos(th),   -b * np.sin(th),     0],
                [b * np.sin(th),        np.cos(th),     0],
                [             0,                 0,     1],
            ])
        def rotx(th, b):
            return np.array([
                [             1,                 0,                   0],
                [             0,        np.cos(th),     -b * np.sin(th)],
                [             0,    b * np.sin(th),          np.cos(th)],
            ])

        edges = [np.zeros(D)] * 4
        # beta = np.arctan(mu * np.pi)
        # R = rotz(beta, 1)
        # f1 = f.dot(R)
        # R = rotz(beta, -1)
        # f2 = f.dot(R)
        # R = rotx(beta, 1)
        # f3 = f.dot(R)
        # R = rotx(beta, -1)
        # f4 = f.dot(R)
        fz = mu * np.linalg.norm(f)
        f1 = f + (mu * np.array([f[0] + fz, f[1] +  0, f[2]]))
        f2 = f + (mu * np.array([f[0] +  0, f[1] + fz, f[2]]))
        f3 = f + (mu * np.array([f[0] - fz, f[1] +  0, f[2]]))
        f4 = f + (mu * np.array([f[0] +  0, f[1] - fz, f[2]]))
        edges = [ 
            f1, 
            f2, 
            f3, 
            f4
        ]
        print("f is:\n{}".format(f))
        print("\nedges:\n{}".format(edges))
        ########## Your code ends here ##########

    else:
        raise RuntimeError("cone_edges(): f must be 3D or 6D. Received a {}D vector.".format(D))

    return edges

def form_closure_program(F):
    """
    Solves a linear program to determine whether the given contact wrenches
    are in form closure.

    Args:
        F - matrix whose columns are 3D or 6D contact wrenches.

    Return:
        True/False - whether the form closure condition is satisfied.
    """
    ########## Your code starts here ##########
    # Hint: you may find np.linalg.matrix_rank(F) helpful
    # TODO: Replace the following program (check the cvxpy documentation)

    # N is the dimension of the wrench space
    # M is the number of wrenches 
    N, M = F.shape 

    # Make sure there are N linearly independent matrices
    r = np.linalg.matrix_rank(F) 
    if r < N: return False

    # k = np.ones((M,))
    # k = cp.Variable()
    # objective = cp.Minimize(k)
    # constraints = [
    #     F*k == 0,
    #     k >= 1
    # ]

    # k = cp.Variable(1)
    k = cp.Variable(F.shape[1])
    # objective = cp.Minimize(k)
    c = np.ones((F.shape[1],))
    # objective = cp.Minimize(c @ k)
    objective = cp.Minimize(c.T @ k)
    # objective = cp.Minimize(cp.sum(cp.sum(F * k , axis = 1)) )
    constraints = [
        # cp.sum(F * k , axis = 0) == 0,
        # np.sum(F * k, axis=1) == 0,
        # cp.sum(cp.sum(F * k , axis = 1)) == 0,
        # F * k == np.zeros((F.shape[0],)),
        F @ k == 0,
        k >= 1
    ]
    # scipy.optimize.linprog(np.ones((F.shape[1],)), A_eq = F, b_eq = np.zeros((F.shape[0], )), bounds = [(1, None) for i in range(F.shape[1])] )
    ########## Your code ends here ##########

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False, solver=cp.ECOS)

    # TODO: delete me!
    # print("Using matrix: \n{}".format(F))
    # print("Found problem status: {}".format(prob.status))
    # print("\n=====\n{}\n=====".format(prob))
    # print(prob.value)

    return prob.status not in ['infeasible', 'unbounded']

def is_in_form_closure(normals, points):
    """
    Calls form_closure_program() to determine whether the given contact normals
    are in form closure.

    Args:
        normals - list of 2D or 3D contact forces.
        points - list of 2D or 3D contact points.

    Return:
        True/False - whether the forces are in form closure.
    """
    ########## Your code starts here ##########
    L = len(normals)
    assert(L > 0)
    assert(L == len(points))

    # dimensionality is N=3 (for 2D), N=6 (for 3D)
    M = L
    N = len(normals[0])
    if N == 2: N += 1
    else: N *= 2

    F = np.zeros((N, M))
    for i in range(L):
        f, tau = wrench(normals[i], points[i])
        F[:, i] = np.concatenate((f, tau), axis=None)

    # TODO: delete me!
    # print()
    # print(F)
    ########## Your code ends here ##########

    return form_closure_program(F)

def is_in_force_closure(forces, points, friction_coeffs):
    """
    Calls form_closure_program() to determine whether the given contact forces
    are in force closure.

    Args:
        forces - list of 2D or 3D contact forces.
        points - list of 2D or 3D contact points.
        friction_coeffs - list of friction coefficients.

    Return:
        True/False - whether the forces are in force closure.
    """
    ########## Your code starts here ##########
    L = len(forces)
    assert(L > 0)
    assert(L == len(points))

    # dimensionality is N=3 (for 2D), N=6 (for 3D)
    M = L
    N = len(forces[0])
    if N == 2: N += 1
    else: N *= 2

    F = np.array([])
    for i in range(L):
        edges = cone_edges(forces[i], friction_coeffs[i])
        for edge in edges:
            f, tau = wrench(edge, points[i])
            w = np.array([np.concatenate((f, tau), axis=0)]).T
            if F.shape[0] == 0: F = w
            else: F = np.hstack((F, w))

    ########## Your code ends here ##########

    return form_closure_program(F)
