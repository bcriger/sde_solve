from numpy import sqrt, dot, eye, isscalar
from numpy.linalg import solve
from numpy.random import randn

#TODO: Use a generic template for a solver given a stepper, then use
#lambdas to curry at the bottom.

__all__ = ['platen_15_step', 'e_m_05_step', 'platen_1_step', 
            'im_e_m_05_step', 'im_platen_1_step', 'im_platen_15_step',
            'milstein_1_step', 'im_milstein_1_step']

# _steppers = [platen_15_step, e_m_05_step, platen_1_step, 
#             im_e_m_05_step, im_platen_1_step, im_platen_15_step,
#             milstein_1_step, im_milstein_1_step]

def platen_15_step(t, rho, det_f, stoc_f, dt, dW):
    """
    Numerically integrates non-autonomous vector-valued stochastic
    differential equations with a single Wiener term:

    :math:`d \rho = \mu(t, \rho) dt + \sigma(t, \rho)dW`.

    Uses the Platen order 1.5 scheme from page 382 of Kloeden/Platen. 
    Uses the result of exercise 5.2.7/PC-Exercise 1.4.12 to obtain 
    delta-Z (the multiple Ito integral I_{1, 0}) given the Wiener 
    increment delta-W.

    :param t: Value of time
    :type t: float
    :param rho: Current value of the solution to the SDE. Must be 
    accepted by `det_f`, `stoc_f` and `e_cb`, see below.
    :type rho: arbitrary, must support arithmetic operators
    :param det_f: callback function which determines the deterministic 
    contribution, the function :math:`\mu(t, \rho)` above. Must have 
    signature `f(t, rho)`.
    :type det_f: function
    :param stoc_f:callback function which determines the stochastic 
    contribution, the function :math:`\sigma(t, \rho)` above. Must have 
    signature `f(t, rho)`.
    :type stoc_f: function
    :param times: the values of time over which integration is to take 
    place. Uniform spacing is assumed, and no intermediate values of 
    time are used internally.
    :type times: iterable
    :param dWs: the values of the Wiener increment at each value of 
    time, for a given trajectory. Must be generated prior to call. 
    :type dWs: iterable
    :param e_cb: callback function which saves information about the 
    solution `rho` at each timestep. Must have signature 
    `f(t, rho, dW)`. 
    :type e_cb: function
    """
    _, I_00, I_01, I_10, I_11, I_111 = _ito_integrals(dt, dW)
    #Evaluations of DE functions
    det_v  = det_f(t, rho)
    stoc_v = stoc_f(t, rho) 
    det_vp = det_f(t + dt, rho)
    stoc_vp = stoc_f(t + dt, rho)
    #Supporting Values
    u_p, u_m = _upsilons(rho, dt, det_v, stoc_v) 
    det_u_p = det_f(t, u_p)
    det_u_m = det_f(t, u_m)
    stoc_u_p = stoc_f(t, u_p)
    stoc_u_m = stoc_f(t, u_m)
    phi_p, phi_m = _phis(dt, u_p, stoc_u_p)
    #Euler term
    rho += _e_m_term(t, rho, det_f, stoc_f, dt, dW)
    #1/(2 * sqrt(dt)) term
    rho += ((det_u_p - det_u_m) * I_10 +
             (stoc_u_p - stoc_u_m) * I_11) / (2. * sqrt(dt)) 
    #1/dt term
    rho += ((det_vp - det_v) * I_00 +
             (stoc_vp - stoc_v) * I_01) / dt 
    #first 1/(2 dt) term
    rho += ((det_u_p - 2. * det_v + det_u_m) * I_00
             + (stoc_u_p - 2. * stoc_v + stoc_u_m) * I_01) / (2. * dt) 
    #second 1/(2 dt) term
    rho += (stoc_f(t, phi_p) - stoc_f(t, phi_m) 
                - stoc_u_p + stoc_u_m) * I_111 / (2. * dt) 

    return rho

def e_m_05_step(t, rho, det_f, stoc_f, dt, dW):
    """
    Numerically integrates non-autonomous vector-valued stochastic
    differential equations with a single Wiener term:

    :math:`d \rho = \mu(t, \rho) dt + \sigma(t, \rho)dW`.

    Uses the Euler-Maruyama order 0.5 scheme from page 341 of 
    Kloeden/Platen. 
    
    :param rho_init: Initial value of the solution to the SDE. Must be 
    accepted by `det_f`, `stoc_f` and `e_cb`, see below.
    :type rho_init: arbitrary
    :param det_f: callback function which determines the deterministic 
    contribution, the function :math:`\mu(t, \rho)` above. Must have 
    signature `f(t, rho)`.
    :type det_f: function
    :param stoc_f:callback function which determines the stochastic 
    contribution, the function :math:`\sigma(t, \rho)` above. Must have 
    signature `f(t, rho)`.
    :type stoc_f: function
    :param times: the values of time over which integration is to take 
    place. Uniform spacing is assumed, and no intermediate values of 
    time are used internally.
    :type times: iterable
    :param dWs: the values of the Wiener increment at each value of 
    time, for a given trajectory. Must be generated prior to call. 
    :type dWs: iterable
    :param e_cb: callback function which saves information about the 
    solution `rho` at each timestep. Must have signature 
    `f(t, rho, dW)`. 
    :type e_cb: function
    """

    rho += _e_m_term(t, rho, det_f, stoc_f, dt, dW) 
    return rho

def platen_1_step(t, rho, det_f, stoc_f, dt, dW):
    """
    Explicit strong order-1 Runge-Kutta, page 376.
    """
    _, _, _, _, I_11, _ = _ito_integrals(dt, dW)
    
    det_v = det_f(t, rho)
    stoc_v = stoc_f(t, rho)
    
    upsilon, _ = _upsilons(rho, dt, det_v, stoc_v)
    
    rho += _e_m_term(t, rho, det_f, stoc_f, dt, dW)
    rho += (stoc_f(t, upsilon) - stoc_v) * I_11 / sqrt(dt)
    
    return rho

def im_e_m_05_step(t, rho, mat_now, mat_fut, stoc_f, dt, dW, alpha=0.5):
    """
    Implicit Euler-Maruyama, page 396.
    """
    det_v = dot(mat_now, rho)
    
    rho += _e_m_term(t, rho, None, stoc_f, dt, dW,
                        alpha=alpha, det_v=det_v)
    
    rho = _implicit_corr(rho, mat_fut, dt, alpha)

    return rho

def im_platen_1_step(t, rho, mat_now, mat_fut, stoc_f, dt, dW, alpha=0.5):
    """
    Implicit strong order-1 Runge-Kutta, page 407.
    """
    _, _, _, _, I_11, _ = _ito_integrals(dt, dW)
    
    det_v = dot(mat_now, rho)
    stoc_v = stoc_f(t, rho)
    
    upsilon, _ = _upsilons(rho, dt, det_v, stoc_v)
    
    rho += _e_m_term(t, rho, None, stoc_f, dt, dW, alpha, det_v)
    
    rho += (stoc_f(t, upsilon) - stoc_v) * I_11 / sqrt(dt)
    
    rho = _implicit_corr(rho, mat_fut, dt, alpha)

    return rho

def im_platen_15_step(t, rho, mat_now, mat_fut, stoc_f, dt, dW,
                        alpha=0.5, return_dZ=False):
    """
    Implicit strong order-1.5 Runge-Kutta.
    """
    _, _, I_01, I_10, I_11, I_111 = _ito_integrals(dt, dW)
    
    #Evaluations of DE functions
    det_v  = dot(mat_now, rho)
    stoc_v = stoc_f(t, rho)  
    stoc_vp = stoc_f(t + dt, rho) 
    
    #Supporting Values
    u_p, u_m = _upsilons(rho, dt, det_v, stoc_v)
    det_u_p = dot(mat_now, u_p) 
    det_u_m = dot(mat_now, u_m) 
    stoc_u_p = stoc_f(t, u_p)
    stoc_u_m = stoc_f(t, u_m)
    phi_p , phi_m = _phis(dt, u_p, stoc_u_p)
    stoc_phi_p = stoc_f(t, phi_p) 
    stoc_phi_m = stoc_f(t, phi_m) 
    
    #Euler term
    rho += _e_m_term(t, rho, None, stoc_f, dt, dW, alpha, det_v) 
    
    #1/dt term
    rho += (stoc_vp - stoc_v) * I_01 / dt 
    rho += 0.5 * (stoc_u_p - 2. * stoc_v + stoc_u_m) * I_01 / dt
    rho += 0.5 * (stoc_phi_p - stoc_phi_m - stoc_u_p + stoc_u_m) * I_111 / dt
    
    #1/sqrt(dt) term
    rho += 0.5 * (det_u_p - det_u_m) * (I_10 - 0.5 * dW * dt) / sqrt(dt)
    rho += 0.5 * (stoc_u_p - stoc_u_m) * I_11 / sqrt(dt)

    rho = _implicit_corr(rho, mat_fut, dt, alpha)
    
    if return_dZ:
        return rho, dZ
    
    return rho

def milstein_1_step(t, rho, det_f, stoc_f, dt, dW, l1_stoc_f):
    """
    Explicit strong order-1 Taylor scheme.
    """
    _, _, _, _, I_11, _ = _ito_integrals(dt, dW)
    
    l1_stoc_v = l1_stoc_f(t, rho)
    
    rho += _e_m_term(t, rho, det_f, stoc_f, dt, dW)
    rho += l1_stoc_v * I_11 
    
    return rho

def im_milstein_1_step(t, rho, mat_now, mat_fut, stoc_f, dt, dW, l1_stoc_f, alpha=0.5):
    """
    Semi-implicit strong order-1 Taylor scheme.
    """
    _, _, _, _, I_11, _ = _ito_integrals(dt, dW)
    
    det_v = dot(mat_now, rho)
    l1_stoc_v = l1_stoc_f(t, rho)
    
    rho += _e_m_term(t, rho, None, stoc_f, dt, dW, alpha, det_v)
    rho += l1_stoc_v * I_11 
    
    rho = _implicit_corr(rho, mat_fut, dt, alpha)

    return rho

#---------------------Convenience Functions---------------------------#

def _ito_integrals(dt, dW=None):
    """
    For low-order solvers, a small batch of Ito integrals is required,
    from I_00 up to I_111. This function calculates all of them in one 
    place, and returns them in lexical order.
    """
    dW = dW if dW else sqrt(dt) * randn()

    u_1, u_2 = dW/sqrt(dt), randn()
    I_10  = 0.5 * dt**1.5 * (u_1 + u_2 / sqrt(3.)) 
    I_00  = 0.5 * dt**2 
    I_01  = dW * dt - I_10 
    I_11  = 0.5 * (dW**2 - dt) 
    I_111 = 0.5 * (dW**2/3. - dt) * dW 

    return dW, I_00, I_01, I_10, I_11, I_111

def _upsilons(rho, dt, det_v, stoc_v):
    """
    In strong Runge-Kutta schemes for stochastic differential 
    equations, derivatives are estimated using supporting function
    evaluations. These values are some of the arguments to those
    functions.
    """    

    u_p = rho + det_v * dt + stoc_v * sqrt(dt)
    u_m = rho + det_v * dt - stoc_v * sqrt(dt)
    
    return u_p, u_m

def _phis(dt, u_p, stoc_u_p):
    """
    Higher-order Runge-Kutta schemes use these supporting values.
    """
    phi_p = u_p + stoc_u_p * sqrt(dt)
    phi_m = u_p - stoc_u_p * sqrt(dt)
    return phi_p, phi_m

def _e_m_term(t, rho, det_f, stoc_f, dt, dW, alpha=0., det_v=None):
    """
    Many stochastic schemes begin with the Euler-Maruyama term, adding 
    higher-order terms afterward. For this reason, we include the 
    Euler-Maruyama term as a convenience function. An optional 
    parameter alpha gives the degree of implicitness; alpha == 0 
    corresponding to a fully explicit scheme, and alpha == 1 being 
    fully implicit.
    """
    if det_v is None:
        det_v = det_f(t, rho)
    return (1. - alpha) * det_v * dt + stoc_f(t, rho) * dW

def _implicit_corr(rho, mat_fut, dt, alpha):
    """
    Implicit steppers solve a vector-valued equation at every time 
    step. Regardless of the order of the stepper, this function depends
    on the current value of rho, the matrix-valued future drift term 
    (the implicit steppers are only set up to work with linear drift
    terms), the timestep, and the implicitness parameter alpha. I 
    include it here as a convenience function.
    """
    #square matrix assumed; but only square matrices make sense.
    if isscalar(mat_fut):
        return rho / (1 - alpha * dt * mat_fut)
    else:
        id_mat = eye(mat_fut.shape[0], dtype=mat_fut.dtype)
        return solve(id_mat - alpha * dt * mat_fut, rho)