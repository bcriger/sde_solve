#sde_solve
###a tiny library for numerically solving stochastic differential equations

##Installation
 + Clone this repo
 + enter the resulting directory
 + `sudo python setup.py install` or `python setup.py install --user` if you don't have superuser privileges. 

##Use
The purpose of this library is to produce numerical solutions to differential equations of the form:

    d_rho = det_f(t,rho) dt + stoc_f(t, rho) dW
using the order-1.5 strong solver for non-autonomous equations from Kloeden and Platen, page 379.

There's only one function in this repo, `sde_platen_15`:

    sde_platen_15(rho_init, det_f, stoc_f, times, dWs, e_cb)

The input variables are:
 + `rho_init`: an initial value. This can be any object that supports arithmetic methods in principle, but it must be accepted as input by `det_f` and `stoc_f`, see below.
 + `det_f`: the deterministic contribution to the derivative. This is a function with signature `det_f(t, rho)`
 + `stoc_f`: the stochastic contribution to the derivative. This is a function with signature `stoc_f(t, rho)`.
 + `times`: values of time (in arb. units) at which the solution is to be calculated. In principle, can be any type which supports `__getitem__`. In practice, NumPy arrays work best. *Note*: As of the current version, `sde_platen_15` assumes this iterable to have a fixed difference.
 + `dWs`: iterable of values for the Wiener increment. This is not generated within the function, since it is often convenient for this variable to be in scope where `sde_platen_15` is called. 
 + `e_cb`: Callback function with signature `e_cb(t, rho, dW)`. This brings the interim values of time, solution and Wiener increment into whatever namespace in which `e_cb` is `def`'d.

##Example
Coming soon

##Future Work
 + Adaptive timesteps
 + Switching the whole thing over to Julia, so I can decide at runtime whether to use an implicit method.