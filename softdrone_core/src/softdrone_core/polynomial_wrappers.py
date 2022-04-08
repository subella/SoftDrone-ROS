import numpy as np
from softdrone_core.msg import MultiPolynomialTrajectory
from softdrone_core.polynomial_trajectory_optimizer import polyder_coeffs


def multi_poly_trajectory_to_wrapper(msg):
    """ Turn a MultiPolynomialTrajectory message into a MultiPolynomialWrapper """
    order = msg.poly_order
    coeffs_x = msg.coeffs_x
    coeffs_y = msg.coeffs_y
    coeffs_z = msg.coeffs_z
    t_list = msg.segment_times
    t_start_global = msg.time_start
    return MultiPolynomialWrapper(order, coeffs_x, coeffs_y, coeffs_z, t_list, speedup=1, t_start_global=t_start_global)


def evaluate_path_time(full_coeffs, T_list, t, speedup, order, der_order):
    T_list = list(T_list)
    cumsum = np.cumsum([0] + T_list + [np.inf])
    t = t*speedup
    seg_index = np.where(cumsum > t)[0][0] - 1
    seg_index = min(seg_index, len(T_list)-1)

    coeffs = full_coeffs[seg_index*(order+1):(seg_index+1)*(order+1)]
    if der_order > 0:
        coeffs_der = (coeffs * polyder_coeffs(order, der_order))
        coeffs_der = coeffs_der[der_order:]
    else:
        coeffs_der = coeffs
    #y = np.polynomial.polynomial.polyval(t - cumsum[seg_index], coeffs_der)
    y = 0
    t_rel = t - cumsum[seg_index]
    for ix in range(len(coeffs_der)):
        y += coeffs_der[ix]*t_rel**ix * speedup**der_order
    return y


def evaluate_path(full_coeffs, T_list, order, der_order, speedup=1, n_pts=50):
    """ Evaluate 1d trajectory at points along path """
    total_time = np.sum(T_list)
    tvals = np.linspace(0, total_time/speedup, n_pts)
    yvals = np.array([evaluate_path_time(full_coeffs, T_list, t, speedup, order, der_order) for t in tvals]).flatten()
    return tvals, yvals


class MultiPolynomialWrapper:
    def __init__(self, order, coeffs_x, coeffs_y, coeffs_z, t_list, speedup, t_start_global=0):
        self.coeffs_x = coeffs_x
        self.coeffs_y = coeffs_y
        self.coeffs_z = coeffs_z
        self.t_list = t_list
        self.speedup = speedup
        self.t_start_global=t_start_global
        self.order = order

        self.t_max = np.sum(t_list)
        self.start_point = self.eval_rel_t(0)
        self.end_point = self.eval_rel_t(self.t_max)

    def eval_rel_t(self, t, der=0, clip_to_tmax=True):
        """ Evaluate path at time t """

        x = evaluate_path_time(self.coeffs_x, self.t_list, t, self.order, der)
        y = evaluate_path_time(self.coeffs_y, self.t_list, t, self.order, der)
        z = evaluate_path_time(self.coeffs_z, self.t_list, t, self.order, der)

        return x,y,z

    def eval_global_t(self, t, der=0, clip_to_tmax=True):
        """ Evaluate path at time t, relative to start t_start_global """
        return self.eval_rel_t(t - self.t_start_global, der=der, clip_to_tmax=clip_to_tmax)

    def eval_full_trajectory(self, der=0):
        tv, xv = evaluate_path(self.coeffs_x, self.t_list, self.order, der, speedup=self.speedup, n_pts=50)
        tv, yv = evaluate_path(self.coeffs_y, self.t_list, self.order, der, speedup=self.speedup, n_pts=50)
        tv, zv = evaluate_path(self.coeffs_z, self.t_list, self.order, der, speedup=self.speedup, n_pts=50)
        return xv, yv, zv

    def eval_global_pva_t(self, t, clip_to_tmax=True):
        """ Evaluate the position, velocity, and acceleration at time t """

        p = self.eval_global_t(t, der=0, clip_to_tmax=True)
        v = self.eval_global_t(t, der=1, clip_to_tmax=True)
        a = self.eval_global_t(t, der=2, clip_to_tmax=True)


        return ( p, v, a )

