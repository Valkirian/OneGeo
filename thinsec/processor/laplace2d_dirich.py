import itertools as it
import numpy as np
import scipy as sp
import scipy.sparse as spsp
import scipy.sparse.linalg as spspla
import pyamg

def laplace2d_rect_accel(lx, ly):

    u0 = laplace2d_rect_analytic(lx, ly, 10)
    boundaries = (np.zeros(ly), np.ones(lx), np.zeros(ly), np.zeros(lx))
    fd_matrix, rhs_vector = laplace2d_rect_fd(lx, ly, boundaries)
    ufull = solve_laplace_system(fd_matrix, rhs_vector, boundaries,
                                 u0[1:-1, 1:-1].flatten())

    return ufull


def laplace2d_rect_analytic(lx, ly, kmax=10):

    xx, yy = np.meshgrid(range(lx), range(ly))
    omega = np.pi / lx
    coef = lambda n: 2*(1 - (-1)**n) /(n * np.pi * np.sinh(n * omega * ly))
    term = lambda n: coef(n) * np.sin(n * omega * xx) * np.sinh(n * omega*(ly - yy))
    # The series vanishes on even 'n'
    n = lambda k: 2*k - 1

    u = term(n(1))
    for k in range(2, kmax):
        u += term(n(k))

    return u


def laplace2d_rect_fd(lx, ly, boundaries=[[], [], [], []]):

    b_left, b_top, b_right, b_bottom = boundaries
    assert len(b_left) == ly, "Left boundary values incomplete"
    assert len(b_right) == ly, "Right boundary values incomplete"
    assert len(b_top) == lx, "Top boundary values incomplete"
    assert len(b_bottom) == lx, "Bottom boundary values incomplete"

    # Generation of finite differences matrix
    cols = lx - 2
    rows = ly - 2
    fd_matrix = laplace_matrix(rows, cols)

    # Generation of right-hand-side vector
    rhs_vector = np.zeros((cols * rows,))
    rhs_vector[::cols] += b_left[1:-1]
    rhs_vector[:cols] += b_top[1:-1]
    rhs_vector[rows-1::cols] += b_right[1:-1]
    rhs_vector[-cols:] += b_bottom[1:-1]

    return fd_matrix, rhs_vector


def laplace_matrix(rows, cols):

    main_block = spsp.diags([-np.ones((cols-1,)),
                             -np.ones((cols-1,)),
                             4*np.ones((cols,))], (-1, 1, 0), (cols, cols))
    side_block = -spsp.eye(cols)
    block_row_parts = [side_block, main_block, side_block] + (rows - 2)*[None,]
    factory = it.cycle(block_row_parts)
    factory.next()
    block_spec = [ [factory.next() for _ in range(rows)] for _ in range(rows) ]
    fd_matrix = spsp.bmat(block_spec, 'bsr')

    return fd_matrix


def solve_laplace_system(fd_matrix, rhs_vector, boundaries, u0=None):

    b_left, b_top, b_right, b_bottom = boundaries
    ly = len(b_left)
    lx = len(b_top)

    uiter, iterinfo = spspla.cg(fd_matrix, rhs_vector, u0)
    ufull = np.empty((ly, lx))
    ufull[0, :] = b_top
    ufull[:, 0] = b_left
    ufull[ly-1, :] = b_bottom
    ufull[:, lx-1] = b_right
    ufull[1:-1, 1:-1] = uiter.reshape((ly-2, lx-2))

    return ufull


def laplace2d_rect_vec(lx, ly, nmax=10):

    xx, yy, nn = np.meshgrid(range(lx), range(ly), range(nmax))
    omega = np.pi / lx
    coef = 4 /((2*nn + 1) * np.pi * np.sinh((2*nn + 1) * omega * ly))
    u = coef * np.sin((2*nn + 1) * omega * xx) * np.sinh((2*nn + 1) * omega*(ly - yy))
    return u.sum(axis=2)


def laplace2d_constant_sides(lx, ly, on_sides=[False, True, False, False]):

    on_left, on_top, on_right, on_bottom = on_sides

    if on_top or on_bottom:
        vertical_sol = laplace2d_rect_accel(lx, ly)

    if on_left or on_right:
        horiz_sol = laplace2d_rect_accel(ly, lx).T

    solution = np.zeros((ly, lx))
    if on_left:
        solution += horiz_sol
    if on_right:
        solution += horiz_sol[:, ::-1]
    if on_top:
        solution += vertical_sol
    if on_bottom:
        solution += vertical_sol[::-1, :]

    return solution


class CornerDirichlet(object):

    def __init__(self, lx, ly, cx, cy):

        assert 1 < cx < lx
        assert 1 < cy < ly

        i_size = (cy - 1, lx - cx - 1)
        ii_size = (ly - cy - 1, cx - 1)
        iii_size = (ly - cy - 1, lx - cx - 1)

        self.cx, self.cy, self.lx, self.ly = cx, cy, lx, ly
        self.i_size, self.ii_size, self.iii_size = i_size, ii_size, iii_size

        i_entries_vec = [ (1+i, cx+j) for i in range(i_size[0])
                                    for j in range(i_size[1]) ]
        ii_entries_vec = [ (cy+i, 1+j) for i in range(ii_size[0])
                                    for j in range(ii_size[1]) ]
        iii_entries_vec = [ (cy+i, cx+j) for i in range(iii_size[0])
                                        for j in range(iii_size[1]) ]
        i_iii_entries = i_entries_vec[-i_size[1]:]
        ii_iii_entries = ii_entries_vec[ii_size[1]-1::ii_size[1]]
        entries_vec = i_entries_vec + ii_entries_vec + iii_entries_vec
        n = len(entries_vec)

        # Using a dict for index inverse lookup is orders of magnitude faster than
        # invoking a list's index() method
        entries_dic = { entry: i for i, entry in enumerate(entries_vec) }

        # Make Finite-Difference discretization system matrix
        i_main = laplace_matrix(*i_size)
        ii_main = laplace_matrix(*ii_size)
        iii_main = laplace_matrix(*iii_size)

        diags = spsp.block_diag((i_main, ii_main, iii_main), 'csr')

        border_entries = i_iii_entries
        compl_entries = [(be[0] + 1, be[1]) for be in border_entries]
        _row = [entries_dic[e] for e in border_entries + compl_entries]
        _col = [entries_dic[e] for e in compl_entries + border_entries]
        _data = [-1,]*len(_row)
        i_iii_block = spsp.coo_matrix((_data, (_row, _col)), shape=(n, n)).tocsr()

        border_entries = ii_iii_entries
        compl_entries = [(be[0], be[1] + 1) for be in border_entries]
        _row = [entries_dic[e] for e in border_entries + compl_entries]
        _col = [entries_dic[e] for e in compl_entries + border_entries]
        _data = [-1,]*len(_row)
        ii_iii_block = spsp.coo_matrix((_data, (_row, _col)), shape=(n, n)).tocsr()

        self.fd_matrix = diags + i_iii_block + ii_iii_block

    def set_boundaries(self, boundaries=[[1, 1, 0], [1, 1, 0]]):

        b_vertical, b_horiz = boundaries
        b_vert_lef, b_vert_mid, b_vert_rgt = b_vertical
        b_hori_top, b_hori_mid, b_hori_bot = b_horiz

        i_size, ii_size, iii_size = self.i_size, self.ii_size, self.iii_size
        cx, cy, lx, ly = self.cx, self.cy, self.lx, self.ly
        ii_offset_vec = np.prod(i_size)
        iii_offset_vec = ii_offset_vec + np.prod(ii_size)

        # Make RHS vector
        rhs_vector = np.zeros((self.fd_matrix.shape[0],))

        rhs_vector[ii_offset_vec:iii_offset_vec:ii_size[1]] += b_vert_lef
        rhs_vector[0:ii_offset_vec:i_size[1]] += b_vert_mid
        rhs_vector[i_size[1]-1:ii_offset_vec:i_size[1]] += b_vert_rgt
        rhs_vector[iii_offset_vec + iii_size[1]-1::iii_size[1]] += b_vert_rgt

        rhs_vector[0:i_size[1]] += b_hori_top
        rhs_vector[ii_offset_vec:ii_offset_vec + ii_size[1]] += b_hori_mid
        rhs_vector[iii_offset_vec-ii_size[1]:iii_offset_vec] += b_hori_bot
        rhs_vector[-iii_size[1]:] += b_hori_bot

        # Make function to shape solution vector into grid
        def vector_to_grid(solution, corner_fill=1):
            solu_grid = np.empty((ly, lx))
            solu_grid[:cy-1, :cx-1] = corner_fill
            # Boundary edges own their "clockwise-most" corner
            solu_grid[cy-1:ly-1, 0] = b_vert_lef
            solu_grid[:cy-1, cx-1] = b_vert_mid
            solu_grid[1:, -1] = b_vert_rgt
            solu_grid[0, cx:] = b_hori_top
            solu_grid[cy-1, 1:cx] = b_hori_mid
            solu_grid[-1, :lx-1] = b_hori_bot
            solu_grid[1:cy, cx:lx-1] = solution[:ii_offset_vec].reshape(*i_size)
            solu_grid[cy:ly-1, 1:cx] = solution[ii_offset_vec:iii_offset_vec].reshape(*ii_size)
            solu_grid[cy:ly-1, cx:lx-1] = solution[iii_offset_vec:].reshape(*iii_size)
            return solu_grid

        self.vector_to_grid = vector_to_grid
        self.rhs_vector = rhs_vector

    def solve(self):

        A = self.fd_matrix
        b = self.rhs_vector

        # Generate B
        B = np.ones((A.shape[0],1), dtype=A.dtype); BH = B.copy()

        # Random initial guess
        np.random.seed(0)
        x0 = sp.rand(A.shape[0],1)

        # Create solver
        ml = pyamg.smoothed_aggregation_solver(A, B=B, BH=BH,
            strength=('symmetric', {'theta': 0.0}),
            smooth="jacobi",
            improve_candidates=[('block_gauss_seidel',
                                 {'sweep': 'symmetric',
                                  'iterations': 4}),] + 14*[None,],
            aggregate="standard",
            presmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
            postsmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
            max_levels=15,
            max_coarse=300,
            coarse_solver="pinv")

        # Solve system
        res = []
        x = ml.solve(b, x0=x0, tol=1e-08, residuals=res,
                     accel="cg", maxiter=300, cycle="V")

        norm_fn = lambda x: pyamg.util.linalg.norm(sp.ravel(b) - sp.ravel(A*x))
        res_rate = (res[-1]/res[0])**(1.0/(len(res)-1.))
        normr0 = norm_fn(x0)

        print " "
        print ml
        print "System size:                " + str(A.shape)
        print "Avg. Resid Reduction:       %1.2f"%res_rate
        print "Iterations:                 %d"%len(res)
        print "Operator Complexity:        %1.2f"%ml.operator_complexity()
        print "Work per DOA:               %1.2f"%(ml.cycle_complexity()/np.abs(sp.log10(res_rate)))
        print "Relative residual norm:     %1.2e"%(norm_fn(x)/normr0)

        residuals = np.array(res)/normr0
        self.solution = x

        return x, residuals

        """
        # Plot residual history
        pylab.semilogy(residuals)
        pylab.title('Residual Histories')
        pylab.xlabel('Iteration')
        pylab.ylabel('Relative Residual Norm')
        pylab.show()
        """

    def get_solution(self, corner_fill=1):

        return self.vector_to_grid(self.solution, corner_fill)

"""
import matplotlib.pyplot as plt
plt.ion()
plt.figure(); plt.imshow(laplace2d_rect(100, 70, 100), cmap=plt.cm.spectral); plt.colorbar()
plt.figure(); plt.imshow( laplace2d_rect(1000, 700, 50) + laplace2d_rect(700, 1000, 50).T, cmap=plt.cm.spectral); plt.colorbar()
"""
"""
mat, rhs, v2s = lp2d.laplace2d_corner_dirichlet(600, 500, 300, 250, [[0, 1, 0], [0, 1, 0]])
plt.matshow(v2s(np.zeros_like(rhs)), cmap=plt.cm.jet); plt.colorbar()
import pyamg
solu_vec = pyamg.solve(mat, rhs, verb=True, tol=1e-8)
plt.matshow(v2s(solu_vec), cmap=plt.cm.jet); plt.colorbar()
"""
