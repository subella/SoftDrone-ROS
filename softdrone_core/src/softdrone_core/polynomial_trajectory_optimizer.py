import casadi as cd
from scipy.special import factorial
import numpy as np
from os import system
import os

def polyder_coeffs_cd(order, k):
    """ polynomial coefficients. low to high order """
    out = cd.MX(1,order + 1)
    for ix in range(order - k + 1):
        out[0,ix + k] = factorial(k + ix)/factorial(ix)
    return out

def polyder_coeffs(order, k):
    """ polynomial coefficients. low to high order """
    out = np.zeros(order + 1)
    for ix in range(order - k + 1):
        out[ix + k] = factorial(k + ix)/factorial(ix)
    return out

def build_Q_block(poly_order, order_to_minimize, T):
    c_j = cd.MX(poly_order + 1,1)
    c_j[order_to_minimize:] = polyder_coeffs_cd(poly_order, order_to_minimize)[order_to_minimize:]
    C = cd.mtimes([c_j, c_j.T])
    T_int = cd.MX(1,poly_order+1)
    T_int[0,order_to_minimize:] = T**(np.arange(poly_order + 1 - order_to_minimize) + 1) / (np.arange(poly_order + 1 - order_to_minimize) + 1)
    T_outer = cd.mtimes([T_int, T_int.T])

    q = T_outer * C
    return q

def build_Q_full(poly_order, order_to_minimize, T, n_segs, q_scale):
    n = (poly_order+1)*n_segs
    qlist = [0]*n_segs
    for ix in range(n_segs):
        for der_ix in range(order_to_minimize):
            qlist[ix] += q_scale[der_ix] * build_Q_block(poly_order, der_ix, T[ix])
    Q = cd.diagcat(*qlist)
    return Q


def build_A_block(poly_order, T):
    # poly_order must be odd
    A = cd.MX(poly_order + 1, poly_order + 1)

    n_derivs_per_point = (poly_order + 1)//2

    for ix in range(n_derivs_per_point):
        row = polyder_coeffs_cd(poly_order, ix)
        A[ix, ix] = row[0,ix]

    for ix in range(n_derivs_per_point):
        row = polyder_coeffs_cd(poly_order, ix)
        times = cd.MX(1,poly_order+1)
        times[0,ix:] = T**np.arange(poly_order + 1 - ix)
        A[ix + n_derivs_per_point, ix:] = (row * times)[ix:]

    return A


def build_A_full(poly_order, T_list):
    n_segs = T_list.size1()
    a_list = [0]*n_segs
    for ix in range(n_segs):
        a_list[ix] = build_A_block(poly_order, T_list[ix])
    A_full = cd.diagcat(*a_list)
    return A_full

#[ A_diag B=0 ]
#[ C      D   ]
#We make use of the Schur-complement, so the inverse is:
#[ inv(A_diag)               0      ]
#[ -inv(D) * C * inv(A_diag) inv(D) ]
def build_A_inv_full(poly_order, T_list):
    n_segs = T_list.size1()
    nd_per_point = (poly_order + 1) // 2
    a_list = [0]*n_segs
    for ix in range(n_segs):
        ai = cd.MX(poly_order + 1, poly_order + 1)
        a = build_A_block(poly_order, T_list[ix])
        A_schur = a[:nd_per_point, :nd_per_point]
        C_schur = a[nd_per_point:, :nd_per_point]
        D_schur = a[nd_per_point:, nd_per_point:]

        A_schur_inv = cd.inv(A_schur)
        D_schur_inv = cd.inv(D_schur)
        ai[:nd_per_point, :nd_per_point] = A_schur_inv
        ai[nd_per_point:, :nd_per_point] = cd.mldivide(D_schur, cd.mtimes([-C_schur, A_schur_inv]))
        ai[nd_per_point:, nd_per_point:] = D_schur_inv
        a_list[ix] = ai

    A_inv = cd.diagcat(*a_list)
    return A_inv


def build_C(poly_order, n_pts):

    ders_per_point = (poly_order + 1)//2
    fixed_ders_per_point = 2
    free_ders_per_point = ders_per_point - fixed_ders_per_point
    fixed_ders_start = ders_per_point
    fixed_ders_end = ders_per_point

    C = cd.MX((poly_order + 1) * (n_pts - 1), (n_pts)*(poly_order + 1)//2)
    n_fixed_constraints = (n_pts - 2) * fixed_ders_per_point + fixed_ders_start + fixed_ders_end
    n_continuity_constraints = (poly_order + 1)//2 * (n_pts - 2)

    for pt_ix in range(n_pts):

        if pt_ix == 0 or pt_ix == n_pts - 1:
            n_duplicate_constraints = 1
        else:
            n_duplicate_constraints = 2
        for offset in range(n_duplicate_constraints):
            fixed_der_index = 0
            free_der_index = 0
            for der in range(ders_per_point):

                # on the first point, need to fix all derivatives
                # on the last point, need to fix all derivatives
                # on other points, need to fix first two derivatives (twice)
                # and free the rest (twice)

                if pt_ix == 0:
                    if der < fixed_ders_start:
                        C[der, der] = 1
                    else:
                        break
                elif pt_ix > 0:
                    # der'th derivative @ pt_ix point
                    row = ders_per_point + (pt_ix-1)*(poly_order + 1) + offset*ders_per_point + der
                    if pt_ix == n_pts - 1:
                        n_fixed = fixed_ders_end
                    else:
                        n_fixed = fixed_ders_per_point
                    if der < n_fixed:
                        # fixed constraint
                        # e.g. for Order 7:
                        # C[4 + (pt_ix - 1)*8 + {0, 1} * 4 + der, 4 + (pt_ix-1)*4 + fixed_der_index]
                        col = fixed_ders_start + (pt_ix-1)*fixed_ders_per_point + fixed_der_index
                        C[row, col] = 1
                        fixed_der_index += 1
                    else:
                        if pt_ix != n_pts - 1:
                            # continuity constriant
                            col = n_fixed_constraints + (pt_ix-1)*free_ders_per_point + free_der_index
                            C[row, col] = 1
                            free_der_index += 1
    return C

def build_opt_coeffs(T_list, df, n_segs, order, order_to_minimize, q_scale_params, extra_q_reg, k_t_scale, j_scale):

    A_full = build_A_full(order, T_list)
    C = build_C(order, n_segs+1)

    A_inv = build_A_inv_full(order, T_list)
    Q = build_Q_full(order, order_to_minimize, T_list, n_segs, q_scale_params)
    Q_reg = Q + extra_q_reg * cd.MX.eye(Q.size1())
    R = cd.mtimes([C.T, A_inv.T, Q_reg, A_inv, C])

    n_fixed_at_endpoint = (order + 1)//2
    n_fixed = (n_segs - 1) * 2 + 2*n_fixed_at_endpoint
    R_ff = R[:n_fixed,:n_fixed]
    R_fp = R[:n_fixed, n_fixed:]
    R_pp = R[n_fixed:, n_fixed:]
    d_p_star = cd.mldivide(R_pp, cd.mtimes([-R_fp.T, df]))
    dsc = cd.vertcat(df, d_p_star)
    opt_coeffs = cd.mldivide(A_full, cd.mtimes([C, dsc]))

    J = j_scale*cd.mtimes([dsc.T, R, dsc]) + k_t_scale * cd.sum1(T_list)
    J_solo = cd.mtimes([dsc.T, R, dsc])
    J_jac = cd.jacobian(J, T_list)

    J_hess = cd.jacobian(J_jac, T_list)

    return opt_coeffs, J, J_solo, J_jac, J_hess

def build_opt_coeffs_solver_3d(n_segs, order, order_to_minimize):

    n_fixed_at_endpoints = (order + 1)//2
    n_fixed = (n_segs - 1) * 2 + 2*n_fixed_at_endpoints
    T_list = cd.MX.sym('T', n_segs)
    df_x = cd.MX.sym('d_fx', n_fixed)
    df_y = cd.MX.sym('d_fy', n_fixed)
    df_z = cd.MX.sym('d_fz', n_fixed)

    q_scale_params = cd.MX.sym('q_scale_param', order_to_minimize+1)
    extra_q_reg = cd.MX.sym('extra_q_reg')
    k_t_scale = cd.MX.sym('k_t_scale')
    j_scale = cd.MX.sym('j_scale')

    opt_coeffs_x, J_x, _, _, _ = build_opt_coeffs(T_list, df_x, n_segs, order, order_to_minimize, q_scale_params, extra_q_reg, k_t_scale, j_scale)
    opt_coeffs_y, J_y, _, _, _ = build_opt_coeffs(T_list, df_y, n_segs, order, order_to_minimize, q_scale_params, extra_q_reg, k_t_scale, j_scale)
    opt_coeffs_z, J_z, _, _, _ = build_opt_coeffs(T_list, df_z, n_segs, order, order_to_minimize, q_scale_params, extra_q_reg, k_t_scale, j_scale)

    J_full = J_x + J_y + J_z + cd.sum1(T_list)
    J_jac = cd.jacobian(J_full, T_list)
    J_hess = cd.jacobian(J_jac, T_list)

    opt_coeffs_solver = cd.Function('opt_coeffs_solver', [T_list, df_x, df_y, df_z, q_scale_params, extra_q_reg, k_t_scale, j_scale], [opt_coeffs_x, opt_coeffs_y, opt_coeffs_z, J_full, J_jac, J_hess])
    return opt_coeffs_solver


def optimize_with_times(n_segs, order, order_to_minimize, ders_fixed, seg_times, q_scale_params, extra_q_reg, k_t_scale, j_scale, opt_coeffs_solver):

    rel_decrease = np.inf
    J_last = np.inf
    tmin = 0.1
    update_max = .2
    ix = 0
    seg_times = np.array(seg_times)
    dfx = ders_fixed[0]
    dfy = ders_fixed[1]
    dfz = ders_fixed[2]
    while 1:
        coeffs_x, coeffs_y, coeffs_z, J, J_jac, J_hess = opt_coeffs_solver(seg_times, dfx, dfy, dfz, q_scale_params, extra_q_reg, k_t_scale, j_scale)

        J_hess = J_hess + .001*cd.DM.eye(J_hess.size1())

        decrease = np.abs(J_last - J)
        rel_decrease = decrease / J
        J_last = J
        if rel_decrease < .01:
            break
        if ix == 100:
            break
        ix += 1

        update_der = np.array(cd.mldivide(J_hess, -J_jac.T))
        update_der[update_der > update_max] = update_max
        update_der[update_der < -update_max] = -update_max
        update_der = update_der.flatten()

        seg_times += update_der
        seg_times = np.array(seg_times)
        seg_times[seg_times < tmin] = tmin

    return np.array(coeffs_x), np.array(coeffs_y), np.array(coeffs_z), seg_times

def generate_3d_solver(n_segs, order, order_to_minimize, output_directory):

    solver = build_opt_coeffs_solver_3d(n_segs, order, order_to_minimize)

    fn_base = 'polyopt_mx_3d_order%d_segs%d' % (order, n_segs)
    file_out_base = os.path.join(output_directory, fn_base)
    c_out_fn = file_out_base + '.c'
    so_out_fn = file_out_base + '.so'
    if not os.path.exists(so_out_fn):
        solver.generate(fn_base + '.c')
        system('mv %s %s; gcc -fPIC -shared -O2 %s -o %s' % (fn_base + '.c', c_out_fn, c_out_fn, so_out_fn)) # Seems to break on -O3

    return so_out_fn

def load_solver(filename):
    f = cd.external('opt_coeffs_solver', filename)
    return f
