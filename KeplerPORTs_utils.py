import numpy as np
import scipy.interpolate as interp
import scipy.special as spec
import scipy.stats as stat
import KeplerPORTs_readers as kpr

"""Module of miscellaneous utility functions for calculating
   planet occurrence rates with Kepler data
"""

def mstar_from_stellarprops(rstar, logg):
    """Gives stellar mass from the rstar and logg
       INPUT:
         rstar - Radius of star [Rsun]
         logg - log surface gravity [cgs]
       OUTPUT:
         mstar - stellar mass [Msun]
    """
    # Convert logg and rstar into stellar mass assuming logg_sun=4.437
    mstar = 10.0**logg * rstar**2. / 10.0**4.437
    return mstar

def transit_duration(rstar, logg, per, ecc):
    """Transit duration
       assuming uniform distribution of cos(inc) orbits,
       assuming rstar/a is small, and assuming rp/rstar is small.
       INPUT:
        rstar - Radius of star [Rsun]
        logg - log surface gravity [cgs]
        per - Period of orbit [day]
        ecc - Eccentricity; hardcoded to be < 0.99 
       OUTPUT:
        durat - Transit duration [hr]
       COMMENTS:  example:  x=transit_duration(1.0,4.437,365.25,0.0)
                            x=10.19559 [hr] duration for planet in 1 year orbit
                            around sun
    """
    # Replace large ecc values with 0.99
    ecc = np.where(ecc > 0.99, 0.99, ecc)
    # Convert logg and rstar into stellar mass
    mstar = mstar_from_stellarprops(rstar, logg)
    # Semi-major axis in AU
    semia = mstar**(1.0/3.0) * (per/365.25)**(2.0/3.0)
    # transit duration e=0 including pi/4 effect of cos(inc) dist
    r_sun = 6.9598e10 # cm
    au2cm = 1.49598e13 # 1 AU = 1.49598e13 cm
    durat = (per*24.0) / 4.0 * (rstar*r_sun) / (semia*au2cm)
    #transit duration e > 0
    durat = durat * np.sqrt(1.0-ecc**2);

    return durat

def calc_density(rstar, mstar):
    """From rstar & mstar calculate average density [cgs]
       INPUT:
         rstar - Radius of star [Rsun]
         mstar - Mass of star [Msun]
       OUTPUT:
         density - Average density [g/cm^3]
    """
    # Assuming density sun = 1.408
    density = 1.408 * mstar / rstar**3.0
    return density

def calc_logg(rstar, mstar):
    """From rstar & mstar calculate logg [cgs]
       INPUT:
         rstar - Radius of star [Rsun]
         mstar - Mass of star [Msun]
       OUTPUT:
         logg - log surface gravity [g/cm^3]
    """
    # Assuming logg sun = 4.437
    logg = 4.437 + np.log10(mstar) - (2.0*np.log10(rstar))
    return logg

def depth_to_rp(rstar, depth):
    """Gives Planet radius [Rear] for a given depth of transit and rstar
       INPUT:
         rstar - Radius of star [Rsun]
         depth - Depth of transit [ppm]
       OUTPUT:
         rp - Radius of planet ***[Rear]***
    """
    # Planet radius assuming Rp=1 Rear and Rstar=1 Rsun has depth 84 ppm    
    rp = np.sqrt(depth/84.0) * rstar
    return rp

def rp_to_depth(rstar, rp):
    """Gives Planet radius [Rear] for a given depth of transit and rstar
       INPUT:
         rstar - Radius of star [Rsun]
         rp - Radius of planet ***[Rear]*** 
       OUTPUT:
         depth - Depth of transit [ppm]
    """
    # Depth of transit assuming Rp=1 Rear and Rstar=1 Rsun has depth 84 ppm    
    depth = 84.0 * rp**2.0 / rstar**2
    return depth

def rp_to_tpssquaredepth(rstar, rp, version=1):
    """Gives average transit depth for a given star and planet radius.
       This simulates the depth of the TPS boxcar signal.
       INPUT:
         rstar - Radius of star [Rsun]
         rp - Radius of planet ***[Rear]***
         version - version = 1 for Q1Q16  
                   version = 2 for Q1Q17***
                       The version 2 result still needs analysis to confirm
       OUTPUT:
         depth - Depth of transit [ppm]
       COMMENTS:
         This function converts the physical rplanet radius
         into the Square pules TPS match filter depth.
         It first converts the k=rp/rstar ratio into the average
         expected limb darkened transit depth at midpoint.
         Then scales the midpoint depth to the average depth using
         results of the Q4 injection run Christiansen et al.(2015)
         Analysis was in /home/cjburke/complete/depth2rp
         Determined that for linear limb darkening =0.6,
         the average limb darkened transit depth,D,
         to purely geometric radius ratio depth, k^2=(rp/rs)^2
         is a roughly linear function of radius ratio.
         D/k^2=alp-bet*k . 
         For u=0.6  alp=1.0874  bet=1.0187
         u=0.7 alp=1.1068 bet=1.0379
         u=0.5 alp=1.0696 bet=1.001
         ***Currently hardcoded for the u=0.6 case***
         Analysis in /home/cjburke/tcert/Q16ops/traninject
         Provides the averaged over transit depth relative to 
         min transit depth
         REALDEPTH2TPSSQUARE=0.84;
    """
    rearthdrsun = 6378137.0 / 696000000.0
    k = rp / rstar * rearthdrsun
    alp = 1.0874
    bet = 1.0187
    depth = (alp-bet*k) * k**2 * 1.0e6
    if version == 1:
        REALDEPTH2TPSSQUARE = 0.84
    elif version == 2:
        REALDEPTH2TPSSQUARE = 1.0
    else:
        REALDEPTH2TPSSQUARE = 0.84
    depth = depth * REALDEPTH2TPSSQUARE
    return depth

def earthflux_at_period(rstar, logg, teff, period):
    """Gives equivalent solar-earth bolometric flux for a given period
       INPUT:
         rstar - Radius of star [Rsun]
         logg - log surface gravity [cgs]
         teff - Effective Temperature [K]
         period - Orbital Period [day]
       OUTPUT:
         flx - Flux relative to sun-earth 
    """
    mstar = mstar_from_stellarprops(rstar, logg)
    # Calculate semi-major axis [AU]
    semia = mstar**(1.0/3.0) * (period/365.25)**(2.0/3.0)
    # Star bolometric luminosity in Lsun assuming teff_sun=5778.0
    lumstar = rstar**2.0 * (teff/5778.0)**4.0
    # Calculate solar earth bolometric flux ratio
    flx = lumstar / semia**2.0
    return flx

def period_at_earthflux(rstar, logg, teff, seff):
    """Gives period for a given equivalent solar-earth bolometric flux
       INPUT:
         rstar - Radius of star [Rsun]
         logg - log surface gravity [cgs]
         teff - Effective Temperature [K]
         seff - insolation flux relative to sun-earth flux
       OUTPUT:
         period - Orbital period [days]
    """
    mstar = mstar_from_stellarprops(rstar, logg)
    # Calculate semi-major axis [AU] assuming teff_sun=5778.0
    semia = rstar * (teff/5778.0)**2 / np.sqrt(seff)
    period = ( semia / (mstar**(1.0/3.0)) )**(3.0/2.0) * 365.25
    return period

def period_from_teq(rstar, logg, teff, teq, f, alb):
    """Gives period that corresponds to an input equillibrium temp
       INPUT:
         rstar - Radius of star [Rsun]
         logg - log surface gravity [cgs]
         teff - Effective Temperature [K]
         teq  - Equillibrium Temperature [K]
         f - Redistribution parameter [1 or 2]
         alb - Albedo 
       OUTPUT:
         teqper - Period of orbit for teq [day]
    """
    # Convert logg and rstar into stellar mass
    mstar = mstar_from_stellarprops(rstar, logg)
    # Calculate semi-major axis [AU] to reach teq
    r_sun = 6.9598e10  #cm
    au2cm = 1.49598e13 # 1 AU = 1.49598e13 cm
    conv = r_sun / au2cm
    semia = 0.5 * (teff/teq)**2  * rstar * np.sqrt(f*(1-alb)) * conv
    # Calculate period of orbit [day]
    teqper = ( semia / (mstar**(1.0/3.0)) )**(3.0/2.0) * 365.25
    return teqper

def teq_from_period(rstar, logg, teff, period, f, alb):
    """Gives equillibrium temperature that corresponds to an input period
       INPUT:
         rstar - Radius of star [Rsun]
         logg - log surface gravity [cgs]
         teff - Effective Temperature [K]
         period  - Period of orbit [day]
         f - Redistribution parameter [1 or 2]
         alb - Albedo 
       OUTPUT:
         teq - Equillibrium temperature [K]
    """
    # Convert logg and rstar into stellar mass
    mstar = mstar_from_stellarprops(rstar, logg)
    # Semi-major axis in AU
    semia = mstar**(1.0/3.0) * (period/365.25)**(2.0/3.0)

    # Calculate Teq [K]
    r_sun = 6.9598e10  #cm
    au2cm = 1.49598e13 # 1 AU = 1.49598e13 cm
    conv = r_sun / au2cm
    teq = teff * np.sqrt(rstar*conv/2.0/semia) * (f*(1.0-alb))**0.25;
    return teq



def interp_trandur(pulses, cdpps, durs):
    """Use pchip interpolation to interpolate
       pulses and cdpps vector.  For values outside
       the domain of pulses return the end values
    """
    pchipobject = interp.PchipInterpolator(pulses, cdpps)
    x = pchipobject(durs)
    # replace results with larger durs than pulses with last cdpps
    x = np.where(durs > pulses[-1], cdpps[-1], x)
    # replace results with smaller durs than pulses with first cddps
    x = np.where(durs < pulses[0], cdpps[0], x)
    # Check for non finite results
    # very very rarely pchip gets NaN results
    # In this case use interp1d with linear interpolation
    # should be more stable
    if np.logical_not(np.all(np.isfinite(x))):
        interpobject = interp.interp1d(pulses, cdpps, kind='linear',
                                       bounds_error=False,
                                       fill_value=0.0)
        x = interpobject(durs)
        x = np.where(durs > pulses[-1], cdpps[-1], x)
        x = np.where(durs < pulses[0], cdpps[0], x)
    return x

def prob_to_transit(rstar, logg, per, ecc):
    """Provides probability to transit for fixed eccentricity.
        assumes uniform distribution on cos(inc) orbits.
        ecc is forced to be < 0.99
       INPUT:
         rstar - Radius of star [Rsun]
         logg - log surface gravity [cgs]
         per - Period of orbit [day]
         ecc - Eccentricity 
       OUTPUT:
         prob - probability to transit
    """
    # Replace large ecc values with 0.99
    ecc = np.where(ecc > 0.99, 0.99, ecc)
    # Convert logg and rstar into stellar mass
    mstar = mstar_from_stellarprops(rstar, logg)
    # Semi-major axis in AU
    semia = mstar**(1.0/3.0) * (per/365.25)**(2.0/3.0)
    # probability to transit e=0 including pi/4 effect of cos(inc) dist
    r_sun = 6.9598e10 # cm
    au2cm = 1.49598e13 # 1 AU = 1.49598e13 cm
    prob = (rstar*r_sun) / (semia*au2cm)
    # prob to transit e > 0
    prob = prob / (1.0-ecc**2)
    # Check for cases where prob > 1.0
    prob = np.where(prob > 1.0, 1.0, prob)
    return prob


def detection_efficiency(mes, mesthresh, version):
    """Gives the Kepler pipeline detection efficiency for a given
       MES.  In other words whats the propability that a transit
       signal of a given MES is recovered by the kepler pipeline.
       There are a variety of functional forms, use the version
       argument to select the one you want.
       INPUT:
         mes - Multiple even statistic
         mesthresh - Mes threshold.  Normally 7.1 unless TPS
                     times out at higher value
         version - Integer specifying functional form
             0 = Standard Theory prob=0.5 at MES=7.1 follows erf
             1 = Fressin et al. (2013) linear ramp 6<MES<16
             2 = Christiansen et al. (2015) gamma distribution
                   for Q1-Q16 pipeline run
       OUTPUT:
         prob - probability for detection [0.0-1.0]
    """
    # Each version has some hard coded parameters
    parm1s = [0.0, 6.0, 4.65]
    parm2s = [1.0, 16.0, 0.98] 
    # Get parameters for specified version
    p1 = parm1s[version]
    p2 = parm2s[version]
    # Do verion 0 which is erf form
    if version == 0:
        muoffset = p1
        sig = p2
        prob = np.full_like(mes, 1.0)
        abssnrdiff = np.abs(mes - mesthresh - muoffset);
        prob = np.where(abssnrdiff < 9.0, \
                        0.5 + (0.5*spec.erf( \
                        abssnrdiff / np.sqrt(2.0*sig**2))),\
                        prob)
        prob = np.where(mes < (mesthresh + muoffset), 1.0 - prob, prob)
    # Do version 1 which is linear ramp
    elif version == 1:
        mesmin = p1
        mesmax = p2
        slope = 1.0 / (mesmax - mesmin)
        prob = (mes - mesmin) * slope
        prob = np.where(prob < 0.0, 0.0, prob)
        prob = np.where(prob > 1.0, 1.0, prob)
    # Do version 2 which is gamma cdf
    elif version == 2:
        a = p1
        b = p2
        usemes = mes - 4.1 - (mesthresh - 7.1)
        usemes = np.where(usemes < 0.0, 0.0, usemes)
        gammawant = stat.gamma(a,loc=0.0,scale=b)
        prob = gammawant.cdf(usemes)
    else:
        prob = np.full_like(mes, 0.0)

    return prob

def floatchoose(m, n):
    """statistical choose function (binomial coefficients)
       that can take float arguments using  gamma functions.
       INPUT:
         m - set of all elements
         n - number of chosen elements
       OUTPUT:
         x - choose function result
    """
    x1 = spec.gammaln(m+1.0)
    x2 = spec.gammaln(n+1.0)
    x3 = spec.gammaln(m-n+1.0)
    x = np.exp(x1 - x2 - x3)
    return x

def floatbino(k, BigM, f):
    """Binomial probability distribution
       allows for floating point values of BigM (number of trials)
       INPUT:
         k - Number of success 
         BigM - Number of trials
         f - single trial success probability
       OUTPUT: 
         x - probability of k successes in BigM trials
    """
    F1 = np.log(floatchoose(BigM,k))
    F2 = k * np.log(f)
    F3 = (BigM-k) * np.log(1.0-f)
    x = np.exp(F1 + F2 + F3)
    return x

def kepler_window_function(period, observe_baseline, dutycycle):
    """Return the kepler window function with the requirement
       of detecting 3 transits.  Use the binomial window function
       approximation from Burke & McCullough (2014).
       INPUT:
         period - Orbital period [day]
         observe_baseline - Observing baseline time from first to
                            last valid observation [day]
         dutycycle - Fraction of data that is valid over observe_baseline
       OUTPUT:
         windowfunc - probability of observing 3 transits
    """
    ntran = observe_baseline / period
    windowfunc = np.full_like(period, 1.0)
    for k in range(3):
        windowfunc = windowfunc - floatbino(k,ntran,dutycycle)
    windowfunc = np.where(windowfunc < 0.0, 0.0, windowfunc)
    return windowfunc

class kepler_single_comp_data:
    """Define a class that contains all the data needed to calculate
       a single target pipeline completeness grid using
       kepler_single_comp()
       CONTENTS:
       id - [int] Target identifier recommend KIC
       period_want - [day] list of orbital periods
       rp_want - [Rearth] list of planet radii
       rstar - [Rsun] star radius
       rstar_p1sig - [Rsun] +1-sigma error interval on star radius
       rstar_n1sig - [Rsun] -1-sigma error interval on star radius
       logg - [cgs] star surface gravity
       teff - [K] stellar effective temperature
       deteffver - [int] detection efficiency version used in call
                     to detection_efficiency()
       ecc - [0.0 - 1.0] orbital eccentricity
       dataspan - [day] scalar observing baseline duration
       dutycycle - [0.0 -1.0] scalar valid data fraction over dataspan
       pulsedurations - [hr] list of transit durations searched
       cdpps - [ppm] cdpp noise at each pulse duration
       mesthresh - [float] mes threshold reached at each pulseduration
                   typically 7.1 for Kepler
       planet_detection_metric_path - [string] directory path
                                        of fits files for the
                                        planet detection metrics
       version - [1 or 2] version 1 is for Q1-Q16 settings
                 version 2 is for Q1-Q17 settings default is 1 
       detefffunc - [function] Flexability to call a different
                    detection efficiency function of your
                    own design.  Defaults to detection_efficiency()
                    of this module
    """
    def __init__(self):
        self.id = 0 
        self.period_want = np.array([0.0])
        self.rp_want = np.array([0.0])
        self.rstar = 0.0
        self.rstar_p1sig = 0.0
        self.rstar_n1sig = 0.0
        self.logg = 0.0
        self.teff = 0.0
        self.deteffver = 0
        self.ecc = 0.0
        self.dataspan = 0.0
        self.dutycycle = 0.0
        self.pulsedurations = np.array([0.0])
        self.cdpps = np.array([0.0])
        self.mesthresh = np.array([0.0])
        self.planet_detection_metric_path = ''
        self.version = 1
        self.detefffunc = detection_efficiency

def kepler_single_comp(data):
    """Calculate a 2D grid of pipeline completeness
       for a single Kepler target.  Follows procedure outlined in
       Burke et al.(2015)
       INPUT:
         data - instance of class kepler_single_comp_data
       OUTPUT:
         probdet - 2D numpy array of period_want vs rp_want
                   pipeline completeness for single target
         probtot - same as probdet, but includes probability to transit
    """

    # Calculate transit duration along period_want list
    transit_duration_1d = transit_duration(data.rstar,
                                           data.logg,
                                           data.period_want,
                                           data.ecc)
    # Calculate interpolated cdpp along transit_duration_1d
    cdpp_1d = interp_trandur(data.pulsedurations,
                             data.cdpps,
                             transit_duration_1d)
    # Calculate interpolated mesthresholds along transit_duration_1d
    mesthresh_1d = interp_trandur(data.pulsedurations,
                                  data.mesthresh,
                                  transit_duration_1d)
    # Make dataspan along transit_duration_1d
    dataspan_1d = np.full_like(transit_duration_1d,data.dataspan)
    # Make dutycycle along transit_duration_1d
    dutycycle_1d = np.full_like(transit_duration_1d,data.dutycycle)
    # get window function along period_want list
    windowfunc_1d = kepler_window_function(data.period_want,
                                           data.dataspan,
                                           data.dutycycle)
    # get geometric probability to transit along period_want list
    probtransit_1d = prob_to_transit(data.rstar,data.logg,
                                     data.period_want,data.ecc)
    # get effective number of transits along period_want list
    ntraneff_1d = (data.dataspan / data.period_want) * data.dutycycle
    # force at least 3 transits available following detection requirement
    ntraneff_1d = np.where(ntraneff_1d < 3.0, 3.0, ntraneff_1d)
    # Calculate transit depth along rp_want list
    depth_1d = rp_to_tpssquaredepth(data.rstar,data.rp_want,data.version)

    # Now ready to make things 2d
    nper = data.period_want.size
    nrp = data.rp_want.size
    depth_2d = np.tile(np.reshape(depth_1d,(nrp,1)),nper)
    cdpp_2d = np.tile(np.reshape(cdpp_1d,(1,nper)),(nrp,1))
    ntraneff_2d = np.tile(np.reshape(ntraneff_1d,(1,nper)),(nrp,1))
    mesthresh_2d = np.tile(np.reshape(mesthresh_1d,(1,nper)),(nrp,1))
    windowfunc_2d = np.tile(np.reshape(windowfunc_1d,(1,nper)),(nrp,1))
    probtransit_2d = np.tile(np.reshape(probtransit_1d,(1,nper)),(nrp,1))


    # Do last calculations
    snr_2d = depth_2d / cdpp_2d * np.sqrt(ntraneff_2d)
    zz_2d = data.detefffunc(snr_2d, mesthresh_2d, data.deteffver)

    probdet = zz_2d * windowfunc_2d
    probtot = probdet * probtransit_2d

    return probdet, probtot

def kepler_single_comp_v1(data):
    """Calculate a 2D grid of pipeline completeness
       for a single Kepler target.  This version has higher fidelity
       since it uses the planet detection metrics fits files
       for the window function and 1-sigma depth function.
       This is much slower than kepler_single_comp()
       INPUT:
         data - instance of class kepler_single_comp_data
       OUTPUT:
         probdet - 2D numpy array of period_want vs rp_want
                   pipeline completeness for single target
         probtot - same as probdet, but includes probability to transit
    """
    # Get the planet detection metrics struct
    print "Read planet detection metrics"
    plan_det_met = kpr.read_planet_detection_metrics(
                         data.planet_detection_metric_path, data.id,
                         want_wf=True,want_osd=True)
    print "Done Reading planet detection metrics"
    # Calculate transit duration along period_want list
    transit_duration_1d = transit_duration(data.rstar,
                                           data.logg,
                                           data.period_want,
                                           data.ecc)
    # To avoid extrapolation force the transit duration to be
    #  within the allowed range
    minduration = data.pulsedurations.min()
    maxduration = data.pulsedurations.max() - 0.01 
    transit_duration_1d = np.where(transit_duration_1d > maxduration,
                                   maxduration, transit_duration_1d)
    transit_duration_1d = np.where(transit_duration_1d < minduration,
                                   minduration, transit_duration_1d)
    # Now go through transit_duration_1d and create an index array
    # into data.pulsedurations such that the index points to nearest
    #  pulse without going over
    pulse_index_1d = np.digitize(transit_duration_1d,data.pulsedurations)
    pulse_index_1d = np.where(pulse_index_1d > 0, pulse_index_1d - 1, pulse_index_1d)
    # Initialize one sigma depth and window function
    one_sigma_depth_1d = np.zeros_like(data.period_want)
    windowfunc_1d = np.zeros_like(data.period_want)
    # iterate over the pulse durations that are present
    for i in range(pulse_index_1d.min(),pulse_index_1d.max()+1):
        idxin = np.arange(data.period_want.size)\
                         [np.nonzero(pulse_index_1d == i)]
        current_period = data.period_want[idxin]
        current_trandur = transit_duration_1d[idxin]
        low_trandur = np.full_like(current_trandur,data.pulsedurations[i])
        hgh_trandur = np.full_like(current_trandur,data.pulsedurations[i+1])
        # Start doing one sigma depth function
        low_osd_x = plan_det_met.osd_data[i]['period']
        low_osd_y = plan_det_met.osd_data[i]['onesigdep']
        low_osd_func = interp.interp1d(low_osd_x, low_osd_y,
                                  kind='nearest', copy=False,
                                  assume_sorted=True)
        tmp_period = np.copy(current_period)
        lowper = low_osd_x[0]
        hghper = low_osd_x[-1]
        tmp_period = np.where(tmp_period < lowper, lowper, tmp_period)
        tmp_period = np.where(tmp_period > hghper, hghper, tmp_period)
        current_low_osd = low_osd_func(tmp_period)
        hgh_osd_x = plan_det_met.osd_data[i+1]['period']
        hgh_osd_y = plan_det_met.osd_data[i+1]['onesigdep']
        hgh_osd_func = interp.interp1d(hgh_osd_x, hgh_osd_y,
                                  kind='nearest', copy=False,
                                  assume_sorted=True)
        tmp_period = np.copy(current_period)
        lowper = hgh_osd_x[0]
        hghper = hgh_osd_x[-1]
        tmp_period = np.where(tmp_period < lowper, lowper, tmp_period)
        tmp_period = np.where(tmp_period > hghper, hghper, tmp_period)
        current_hgh_osd = hgh_osd_func(tmp_period)
        # linear interpolate across pulse durations
        keep_vector = (current_trandur - low_trandur) /\
                      (hgh_trandur - low_trandur)
        current_osd = (current_hgh_osd-current_low_osd) * \
                      keep_vector + current_low_osd
        one_sigma_depth_1d[idxin] = current_osd

        # Start doing window function
        low_wf_x = plan_det_met.wf_data[i]['period']
        low_wf_y = plan_det_met.wf_data[i]['window']
        low_wf_func = interp.interp1d(low_wf_x, low_wf_y,
                                  kind='nearest', copy=False,
                                  assume_sorted=True)
        tmp_period = np.copy(current_period)
        lowper = low_wf_x[0]
        hghper = low_wf_x[-1]
        tmp_period = np.where(tmp_period < lowper, lowper, tmp_period)
        tmp_period = np.where(tmp_period > hghper, hghper, tmp_period)
        current_low_wf = low_wf_func(tmp_period)
        hgh_wf_x = plan_det_met.wf_data[i+1]['period']
        hgh_wf_y = plan_det_met.wf_data[i+1]['window']
        hgh_wf_func = interp.interp1d(hgh_wf_x, hgh_wf_y,
                                  kind='nearest', copy=False,
                                  assume_sorted=True)
        tmp_period = np.copy(current_period)
        lowper = hgh_wf_x[0]
        hghper = hgh_wf_x[-1]
        tmp_period = np.where(tmp_period < lowper, lowper, tmp_period)
        tmp_period = np.where(tmp_period > hghper, hghper, tmp_period)
        current_hgh_wf = hgh_wf_func(tmp_period)
        # linear interpolate across pulse durations
        current_wf = (current_hgh_wf-current_low_wf) * \
                      keep_vector + current_low_wf
        windowfunc_1d[idxin] = current_wf

    # Calculate interpolated mesthresholds along transit_duration_1d
    mesthresh_1d = interp_trandur(data.pulsedurations,
                                  data.mesthresh,
                                  transit_duration_1d)
    # get geometric probability to transit along period_want list
    probtransit_1d = prob_to_transit(data.rstar,data.logg,
                                     data.period_want,data.ecc)
    # Calculate transit depth along rp_want list
    depth_1d = rp_to_tpssquaredepth(data.rstar,data.rp_want,data.version)

    # Now ready to make things 2d
    nper = data.period_want.size
    nrp = data.rp_want.size
    depth_2d = np.tile(np.reshape(depth_1d,(nrp,1)),nper)
    one_sigma_depth_2d = np.tile(np.reshape(one_sigma_depth_1d,
                                 (1,nper)),(nrp,1))
    mesthresh_2d = np.tile(np.reshape(mesthresh_1d,(1,nper)),(nrp,1))
    windowfunc_2d = np.tile(np.reshape(windowfunc_1d,(1,nper)),(nrp,1))
    probtransit_2d = np.tile(np.reshape(probtransit_1d,(1,nper)),(nrp,1))


    # Do last calculations
    snr_2d = depth_2d / one_sigma_depth_2d
    zz_2d = data.detefffunc(snr_2d, mesthresh_2d, data.deteffver)

    probdet = zz_2d * windowfunc_2d
    probtot = probdet * probtransit_2d
    return probdet, probtot
