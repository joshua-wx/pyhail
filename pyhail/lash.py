"""
LASH (Large Accumulations of Severe Hail) sub-module of pyhail

Contains the LASH retrieval for gridded radar data.
Required reflectivity, MESH, HCA (optional), pressure and temperature data.

Joshua Soderholm - 15 June 2018
"""


"""
Inputs:
Corrected and gridded Reflectivity
MESH
gridded HCA
0deg isotherm altitude
mean pressure betweek surface and 0 deg wet-bulb temperature
"""

"""
Steps:
- Calculate iwc_h below 0deg isotherm and where HCA is either hail or rain-hail
- Using MESH, calculate hailstone fall velocity
- Time integration using iwc_h and fall velocity. This should be in a notebook.
"""



#parameters
epsilon = 0.814 # statistical best coefficent, Wallace et al. 2019 fig 9
eta = 0.64 # unitless, packing density of monodisperse spheres (Scott and Kilgour 1969)
rho_h = 0.9 #g/cm3, ice density

#functions
pc = (1000*P**-1)**0.545 #Heymsfield and Wright (2014)
iwc_h = 0.000044*refl_z**0.71 # Heymsfield and Miller (1988)
hail_v = 488*(hail_sz**0.84)*pc #from Heymsfield and Wright (2014) but discussed in Wallace et al. 2019

#integration
hAcc = (1/epsilon)*(1/eta*rho_h)* sum_of(iwc_h*hail_v*delta_t) #Kalina et al. (2016)





"""
References:

Heymsfield, A. J., & Miller, K. M. (1988). Water Vapor and ice Mass Transported into the Anvils Of CCOPE Thunderstorms: Comparison with Storm Influx and Rainout, Journal of Atmospheric Sciences, 45(22), 3501-3514. Retrieved Apr 28, 2021, from https://journals.ametsoc.org/view/journals/atsc/45/22/1520-0469_1988_045_3501_wvaimt_2_0_co_2.xml

Heymsfield, A., & Wright, R. (2014). Graupel and Hail Terminal Velocities: Does a “Supercritical” Reynolds Number Apply?, Journal of the Atmospheric Sciences, 71(9), 3392-3403. Retrieved Apr 28, 2021, from https://journals.ametsoc.org/view/journals/atsc/71/9/jas-d-14-0034.1.xml

Kalina, E. A., Friedrich, K., Motta, B. C., Deierling, W., Stano, G. T., & Rydell, N. N. (2016). Colorado Plowable Hailstorms: Synoptic Weather, Radar, and Lightning Characteristics, Weather and Forecasting, 31(2), 663-693. Retrieved Apr 28, 2021, from https://journals.ametsoc.org/view/journals/wefo/31/2/waf-d-15-0037_1.xml

Scott, G. D., and D. M. Kilgour, 1969: The density of random close packing of spheres. J. Phys. D, 2, 863–866, https://doi.org/10.1088/0022-3727/2/6/311.

Wallace, R., Friedrich, K., Kalina, E. A., & Schlatter, P. (2019). Using Operational Radar to Identify Deep Hail Accumulations from Thunderstorms, Weather and Forecasting, 34(1), 133-150. Retrieved Apr 28, 2021, from https://journals.ametsoc.org/view/journals/wefo/34/1/waf-d-18-0053_1.xml
"""