from numpy import sqrt
from numpy.random import randn

#TODO: Use a generic template for a solver given a stepper, then use lambdas to curry at the bottom.

__all__ = ['sde_platen_15', 'platen_15_step', 'sde_e_m_05', 'e_m_05_step']

def sde_platen_15(rho_init, det_f, stoc_f, times, dWs, e_cb):
    """
    Numerically integrates non-autonomous vector-valued stochastic
    differential equations with a single Wiener term:

    :math:`d \rho = \mu(t, \rho) dt + \sigma(t, \rho)dW`.

    Uses the Platen order 1.5 scheme from page 382 of Kloeden/Platen. 
    Uses the result of exercise 5.2.7/PC-Exercise 1.4.12 to obtain 
    delta-Z (the multiple Ito integral I_{1, 0}) given the Wiener 
    increment delta-W.

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
    dt = times[1] - times[0] #fixed dt assumed

    rho = rho_init
    for idx, t in enumerate(times):
        dW = dWs[idx]
        e_cb(t, rho, dW)
        if not(t == times[-1]):
            rho = platen_15_step(t, rho, det_f, stoc_f, dt, dW)

    pass #subroutine

def platen_15_step(t, rho, det_f, stoc_f, dt, dW):
    """
    Advances rho(t) to rho(t+dt), subject to a stochastic kick of dW.
    """
    _, I_01, I_10, I_11, I_111 = _ito_integrals(dt, dW)
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

def sde_e_m_05(rho_init, det_f, stoc_f, times, dWs, e_cb):
    """
    Numerically integrates non-autonomous vector-valued stochastic
    differential equations with a single Wiener term:

    :math:`d \rho = \mu(t, \rho) dt + \sigma(t, \rho)dW`.

    Uses the Euler-Maruyama order 0.5 scheme from page INSERT PAGE of 
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
    dt = times[1] - times[0] #fixed dt assumed

    rho = rho_init
    for idx, t in enumerate(times):
        dW = dWs[idx]
        e_cb(t, rho, dW)
        if not(t == times[-1]):
            rho = e_m_05_step(t, rho, det_f, stoc_f, dt, dW)

    pass #subroutine

def e_m_05_step(t, rho, det_f, stoc_f, dt, dW):
    """
    Advances rho(t) to rho(t+dt), subject to a stochastic kick of dW.
    """

    rho += _e_m_term(t, rho, det_f, stoc_f, dt, dW) 
    return rho

def sde_platen_1(rho_init, det_f, stoc_f, times, dWs, e_cb):
    """
    Does the same thing as the order-1.5 solver above, but at order 1.
    """
    dt = times[1] - times[0] #fixed dt assumed

    rho = rho_init
    for idx, t in enumerate(times):
        dW = dWs[idx]
        e_cb(t, rho, dW)
        if not(t == times[-1]):
            rho = platen_1_step(t, rho, det_f, stoc_f, dt, dW)

    pass #subroutine    

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
    
    pass #subroutine


#---------------------Convenience Functions---------------------------#

def _ito_integrals(dt, dW=None):
    """
    For low-order solvers, a small batch of Ito integrals is required,
    from I_00 up to I_111. This function calculates all of them in one 
    place, and returns them in lexical order.
    """
    if dW is None:
        dW = np.sqrt(dt) * randn()

    u_1, u_2 = dW/sqrt(dt), randn()
    I_10  = 0.5 * dt**1.5 * (u_1 + u_2/sqrt(3.)) 
    I_00  = 0.5 * dt**2 
    I_01  = dW * dt - I_10 
    I_11  = 0.5 * (dW**2 - dt) 
    I_111 = 0.5 * (dW**2/3. - dt) * dW 

    return dW, I_01, I_10, I_11, I_111

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

def _e_m_term(t, rho, det_f, stoc_f, dt, dW, alpha=0.):
    """
    Many stochastic schemes begin with the Euler-Maruyama term, adding 
    higher-order terms afterward. For this reason, we include the 
    Euler-Maruyama term as a convenience function. An optional 
    parameter alpha gives the degree of implicitness; alpha == 0 
    corresponding to a fully explicit scheme, and alpha == 1 being 
    fully implicit.
    """
    return (1. - alpha) * det_f(t, rho) * dt + stoc_f(t, rho) * dW 