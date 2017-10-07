#helicoil.py
"""
Design of Springs to EN 13906-1:2013.

For hot-coiled coil springs with closed, ground ends, where the 
principal load is axial.

Notation and Units follow the EN spec
"""
import time
from collections import namedtuple
from math import pi, sqrt, tan, atan, cos
from scipy.interpolate import interp1d
from scipy.optimize import fmin, fsolve
import numpy as np


#=============================== Spring Geometry ============================

def spring_index(D, d):
    """
    Return the spring index, 'w'.
    """
    return D/d


def total_coils(n):
    """
    Return the total number of coils, 'n_t'. S9.8
    """
    return n + 1.5


def active_coils(G, d, D, R):
    """
    Return the number of active coils, 'n'. S9.7

    Note that R = F/s. For some reason EN use F and s (!)
    """
    return G * d**4 / (8.0 * R * D**3)


def Sa_min_reserve_length(D, d, n):
    """
    Returns Sa, the min required length beyond solid (static). S9.9
    """
    return 0.02 * n * (D + d)


def solid_length(n, d_max):
    """
    Return the spring solid length, 'Lc'. S9.10
    """
    return (total_coils(n) - 0.3) * d_max


def diameter_swell(D, d, s_c, n):
    """
    Return the increase in outside diameter [mm]. Section 9.11 (19)
    """
    m = (s_c + n*d)/n
    
    return 0.1 * (m**2 - 0.8 * m * d - 0.2 * d**2) / D


def modulus_temp_factor(t, G):
    """
    Returns modulus at the given temperature. Section 8.2, eqn 2.
    """
    r = 0.25e-3  # EN 10089 steels

    return G*(1.0 - r*(t - 20.0))


def coil_mass(D, L0, n_t, d, rho):
    """
    Return the coil mass in kg.
    """
    density = rho*1.0e-6 # convert kg/litre to kg/mm^3
    circumference = pi*D
    swell = cos(atan(L0/(circumference*(n_t + 1.75))))
    barLength = circumference*n_t/swell
    barArea = 0.25*pi*d**2

    return (barLength - 0.4*circumference)*barArea*density


#=============================== Spring Rates ===============================

def axial_rate(G, d, D, n):
    """
    Return spring axial rate, 'R'. S9.4
    """
    return G * d**4 / (8.0 * n * D**3)


def lateral_rate(G, E, d, D, F, R, L):
    """
    Return spring lateral (shear) rate, 'Sy'
    """
    modRatio = G/E

    rigBend = 0.5*R*L*D**2/(1.0 + 2.0*modRatio)
    rigShear = R*L/modRatio

    dis_crt = F*(1.0 + F/rigShear)/rigBend

    if dis_crt > 0.0:
        factor = sqrt(dis_crt)
        dist = (1.0 + F/rigShear)*tan(0.5*factor*L)/factor
        latRate = F/(2.0*dist - L)
    else:
        latRate = 0.0

    return latRate


#=============================== Stresses ===================================

def stress_correction(springIndex):
    """
    Return stress correction factor to Bergstrasser. Fig 3.
    """
    return (springIndex + 0.5)/(springIndex - 0.75)


def axial_stress_static(D, d, load):
    """
    Return shear stress due to axial compression of spring, for static loads.
    """
    return 8.0 * D * load / (pi * d**3)


def axial_stress_dynamic(D, d, F):
    """
    Return shear stress due to axial compression of spring, for dynamic loads.
    """
    return stress_correction(spring_index(D, d)) * axial_stress_static(D, d, F)


def lateral_stress(G, E, d, D, F, R, L, latDefln):
    """
    Return shear stress due to lateral shear of the spring.
    """
    correction = stress_correction(spring_index(D, d))
    modRatio = G/E

    rigBend = 0.5 * R * L * D**2 / (1.0 + 2.0 * modRatio)
    rigShear = R * L / modRatio

    dis_crt = F * (1.0 + F / rigShear) / rigBend

    if dis_crt > 0.0:
        factor = sqrt(dis_crt)
        dist = (1.0 + F/rigShear)*tan(0.5*factor*L)/factor
        latRate = F/(2.0*dist - L)
    else:
        latRate = 0.0
        dist = 0.0

    return 16.0*correction*latRate*latDefln*dist/(pi*d**3)


#=============================== Performance Checks =========================

def buckling_deflection(G, E, D, freeLength, endCond):
    """
    Return the critical deflection from free L at which buckling occurs.

    :Parameters:
        endCond: float. 2.0:free, to 0.5:fully guided.  See EN figure 5
    """
    modRatio = G/E
    Acoeff = 0.5/(1.0 - modRatio)
    Bcoeff = (1.0 - modRatio)/(0.5 + modRatio)
    dis_crt = 1.0 - Bcoeff*(pi*D/(endCond*freeLength))**2

    if dis_crt < 1.0e-9:
        critDefln = freeLength # that is, no buckling at all
    else:
        critDefln = freeLength*Acoeff*(1.0 - sqrt(dis_crt))
        critDefln = min(critDefln, freeLength)

    return critDefln


def fundamental_frequency(n, d, D, G, density):
    """
    Returns frequency [Hz] of fundamental axial vibration mode.

    EN Section 9.12 Eqn 20.
    """
    return 3560.0*d/(n*D**2)*sqrt(G/density)


#=============================== Material Properties ========================

class GoodmanCurves:
    """
    The EN-style Modified Goodman curves for round bar.
    """
    def __init__(self, d, y0, max_stress, knee_stress):
        """
        :Parameters:
            d: Bar diameters
            y0: float. [MPa] Allowable max stress at zero min stress
            max_stress: float [MPa] Maximum allowable stresses
            kneeMins: float [MPa] Min stress value where upper stress limit
                                 stops increasing.
        """
        self.y0 = interp1d(d, y0, fill_value="extrapolate")
        self.max = interp1d(d, max_stress, fill_value="extrapolate")
        self.knee = interp1d(d, knee_stress, fill_value="extrapolate")


    def upper_stress_limit(self, d, minStress):
        """
        Return the allowable maximum stress given a d, min stress.
        """
        if minStress <= 0.0:
            limit = self.y0(d)
        elif minStress >= self.knee(d):
            limit = self.max(d)
        else:
            y0 = self.y0(d)
            limit = y0 + (self.max(d) - y0)*minStress/self.knee(d)

        return limit


    def allowable_range(self, d, minStress):
        """
        Return the allowable stress range given the minimum stress.
        """
        return self.upper_stress_limit(d, minStress) - minStress


    def stress_range_reserve(self, d, minStress, maxStress):
        """
        Return the allowed stress range minus the actual range (can be -ve)
        """
        stressRange = maxStress - minStress
        allowableRange = self.allowable_range(d, minStress)

        return allowableRange - stressRange


##############################################################################
Material = namedtuple('Material', ['name', 'E', 'G', 'rho', 
                      'low_cycle_GC', 'high_cycle_GC', 'solid_stress_limit'])


prEN10089 = Material(
    name = "prEN 10089:2000 special quality steel, ground, shot peened",
    E = 206000.0,
    G = 78500.0,
    rho = 7.85,
    low_cycle_GC = GoodmanCurves(
        [ 10.0,  15.0,  25.0,  35.0,  50.0], # bar diameters
        [760.0, 670.0, 590.0, 515.0, 430.0], # y-intercept
        [890.0, 830.0, 780.0, 740.0, 690.0], # max stress
        [225.0, 262.0, 305.0, 335.0, 393.0]), # knee values
    high_cycle_GC = GoodmanCurves(
        [ 10.0,  15.0,  25.0,  35.0,  50.0], # bar diameters
        [645.0, 555.0, 475.0, 405.0, 325.0], # y-intercept
        [890.0, 830.0, 780.0, 740.0, 690.0], # max stress
        [390.0, 410.0, 440.0, 460.0, 490.0]), # knee values
    solid_stress_limit = interp1d(
        [  7.5,  10.0,  12.5,  15.0,  20.0,  25.0,  30.0,  40.0,  50.0,  60.6],
        [955.0, 925.0, 896.9, 874.4, 840.2, 813.1, 794.6, 761.3, 735.8, 716.5],
        fill_value="extrapolate")
    )


def fatigue_stress_reserve(GCs, G, d, D, n, L0, L, deflnAmp):
    """
    Return the reserve of actual versus allowed stress range.
    """
    R = axial_rate(G, d, D, n)

    minLoad = R*(L0 - L - deflnAmp)
    maxLoad = R*(L0 - L + deflnAmp)

    minStr = axial_stress_dynamic(D, d, minLoad)
    maxStr = axial_stress_dynamic(D, d, maxLoad)

    return GCs.stress_range_reserve(d, minStr, maxStr)


def min_fatigue_stress_reserve(mat, d, D, n, L0, L, 
                               lo_cycle_amp, hi_cycle_amp):
    """
    Return the minimum reserve of actual versus allowed stress range.
    """
    lo_cycle_res = fatigue_stress_reserve(mat.low_cycle_GC, mat.G,
                                      d, D, n, L0, L, lo_cycle_amp)
    hi_cycle_res = fatigue_stress_reserve(mat.high_cycle_GC, mat.G,
                                      d, D, n, L0, L, hi_cycle_amp)

    return min(lo_cycle_res, hi_cycle_res)



CoilData = namedtuple("CoilData", """
    name curr_time mat_name mat_E mat_G mat_rho 
    d D w OD ID n n_t
    L0 Lc L_min Sa coil_gap
    L L1 L2
    F F1 F2
    R RQ_0 RQ_1 RQ_2
    end_cons buk_lens
    solid_str solid_str_limit solid_str_reserve
    lo_cycle_amp hi_cycle_amp 
    lo_cycle_res hi_cycle_res
    freq mass
    """)


def coil_data(name, d, D, n, F, L, L_min, lo_cycle_amp, hi_cycle_amp, mat):
    """
    Returns a string (csv) of spring coil information.
    """
    R = axial_rate(mat.G, d, D, n)

    L0 = L + F/R
    Lc = solid_length(n, d)
    
    Sa = Sa_min_reserve_length(D, d, n)  # !!! Relate to L_min
    coil_gap = (L0 - Lc)/n

    solid_str = axial_stress_static(D, d, R*(L0-Lc))
    solid_str_limit = mat.solid_stress_limit(d)

    n_t = total_coils(n)
    
    L1 = L - lo_cycle_amp
    L2 = L + lo_cycle_amp

    F1 = F + lo_cycle_amp*R
    F2 = F - lo_cycle_amp*R

    RQ_0 = lateral_rate(mat.G, mat.E, d, D, F,  R, L)
    RQ_1 = lateral_rate(mat.G, mat.E, d, D, F1, R, L1)
    RQ_2 = lateral_rate(mat.G, mat.E, d, D, F2, R, L2)

    lo_cycle_res = fatigue_stress_reserve(mat.low_cycle_GC, mat.G,
                                    d, D, n, L0, L, lo_cycle_amp)
    hi_cycle_res = fatigue_stress_reserve(mat.high_cycle_GC, mat.G,
                                    d, D, n, L0, L, hi_cycle_amp)

    # Buckling lengths for different end conditions
    end_cons = np.array([2.0, 1.0, 0.7, 0.5])
    buk_lens = np.array([L0 - buckling_deflection(mat.G, mat.E, D, L0, end_con)
                            for end_con in end_cons])

    return CoilData(
        name=name, curr_time=time.asctime(),
        mat_name=mat.name, mat_E=mat.E, mat_G=mat.G, mat_rho=mat.rho,
        d=d, D=D, w=spring_index(D,d), OD=D+d, ID=D-d, n=n, n_t=n_t,
        L0=L0, Lc=Lc, L_min=L_min, Sa=Sa, coil_gap=coil_gap,
        L=L, L1=L1, L2=L2,
        F=F, F1=F1, F2=F2,
        R=R, RQ_0=RQ_0, RQ_1=RQ_1, RQ_2=RQ_2,
        end_cons=end_cons, buk_lens=buk_lens,
        solid_str=solid_str, solid_str_limit=solid_str_limit,
        solid_str_reserve=solid_str_limit - solid_str,
        lo_cycle_amp=lo_cycle_amp, hi_cycle_amp=hi_cycle_amp,
        lo_cycle_res=lo_cycle_res, hi_cycle_res=hi_cycle_res,
        freq=fundamental_frequency(n, d, D, mat.G, mat.rho),
        mass=coil_mass(D, L0, n_t, d, mat.rho)
        )


def list_str(values, fmt="5.1f", sep=","):
    """
    """
    return sep.join([f"{value:{fmt}}" for value in values])


def coil_data_csv(coil):
    """
    Returns a string (csv) of spring coil information.
    """
    # need to format these arrays first; the default has too much precision
    end_con_str = list_str(coil.end_cons)
    buk_len_str = list_str(coil.buk_lens)

    return f"""
Description              ,symbol, units, value
Coil Name                ,      ,      ,{coil.name}
Time                     ,      ,      ,{coil.curr_time}

Material name            ,      ,      ,{coil.mat_name}
Young's Modulus          ,  E   ,  MPa ,{coil.mat_E:7.0f}
Shear Modulus            ,  G   ,  MPa ,{coil.mat_G:7.0f}
Density                  , rho  , kg/l ,{coil.mat_rho:5.2f}

Hot Coiled               ,      ,      ,{True}
Ground Ends              ,      ,      ,{True}
Closed Ends              ,      ,      ,{True}

Design Load,             ,  F   ,  N   ,{coil.F:8.1f}
Design Length            ,  L   ,  mm  ,{coil.L:8.1f}

Mean Coil Diameter       ,  D   ,  mm  ,{coil.D:6.2f}
Bar Diameter             ,  d   ,  mm  ,{coil.d:7.3f}
Spring Index             ,  w   ,      ,{coil.w:7.3f}

Outside Diameter         ,  Do  ,  mm  ,{coil.OD:6.2f}
Inside Diameter          ,  Di  ,  mm  ,{coil.ID:6.2f}

Num Active Coils         ,   n  ,      ,{coil.n:5.2f}
Num Total Coils          , n_t  ,      ,{coil.n_t:5.2f}

Axial Rate               ,  R   , N/mm ,{coil.R:5.1f}
Lateral Rate, nominal    ,  RQ  , N/mm ,{coil.RQ_0:5.1f}, at ,{coil.L:6.1f}, mm length
 at typical min length   ,  RQ  , N/mm ,{coil.RQ_1:5.1f}, at ,{coil.L1:6.1f}, mm length
 at typical max length   ,  RQ  , N/mm ,{coil.RQ_2:5.1f}, at ,{coil.L2:6.1f}, mm length

Free Length              ,  L0  ,  mm  ,{coil.L0:6.1f}
Solid Length             ,  Lc  ,  mm  ,{coil.Lc:6.1f}
Min service len          ,  Lm  ,  mm  ,{coil.L_min:6.1f}
Allowed min len          ,  Ln  ,  mm  ,{coil.Lc+coil.Sa:6.1f}
Gap between coils        ,  a0  ,  mm  ,{coil.coil_gap:6.1f}

Seating coeffs           ,  mu  ,      ,{end_con_str}
Buckling lengths         ,  LK  ,  mm  ,{buk_len_str}

The Fatigue deflection amplitudes (about the design height) are:
Defln amp for 1e5 cycles ,      ,  mm  ,{coil.lo_cycle_amp:5.1f}
Defln amp for 2e6 cycles ,      ,  mm  ,{coil.hi_cycle_amp:5.1f}

Lo cycle fatigue reserve ,      , MPa  ,{coil.lo_cycle_res:6.1f}
Hi cycle fatigue reserve ,      , MPa  ,{coil.hi_cycle_res:6.1f}

Solid stress             ,      , MPa  ,{coil.solid_str:6.1f}
Solid stress reserve     ,      , MPa  ,{coil.solid_str_reserve:6.1f}

Natural Frequency        ,  f0  ,  Hz  ,{coil.freq:5.1f}
Mass                     ,      ,  kg  ,{coil.mass:5.1f}
"""