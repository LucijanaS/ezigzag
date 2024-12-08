import numpy as np

def angdd(z1,z2):
    def f(r,z):
        omm = 0.3
        omr = 1e-4
        oml = 1- omm - omr
        E = (oml + omm*(1+z)**3 + omr*(1+z)**4)**(1/2)
        return (1/E,)
    z = np.linspace(z1,z2,2)
    integral = odeint(f,(0,),z)
    r = integral[:,-1] / (1+z2)
    return r

c = 299792458
au = 149597870700

arcsec = np.pi/3 * 60**(-3)
pc = au/arcsec
ch100 = c * 1e5/(1e6*pc)
Gmsol = 1.3271244e20

if __name__ == '__main__':
    print('100 km/s/Mpc = %7.2e m/s^2' % ch100)
    print('GMsol/pc**2 = %7.2e m/s^2' % (Gmsol/pc**2))
    
