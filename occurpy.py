import argparse
import numpy as np
import multiprocessing
from scipy.integrate import dblquad
"""Module for calculating the planet occurrence rates following
   the results from Burke et al.(2015).  See the documentation
   for occurpy function below for more details.
   example: 'python occur.py -h' at the command line
               lists positional and optional arguments
               for the command line interface.
   example: 'python occur.py 1.0 2.0 50.0 200.0 -np 4 -v
             calculates the baseline occurrence rate posterior for F_1
             from Burke et al.(2015) using 4 processes to speed up the
             calculation and prints out verbose status messages 
"""

def singlepbroker(rp, per, alp1, alp2, rpbrk, bet, rmin, rmax,
                  pmin, pmax, normalized):
    """Broken powerlaw in Rp shape function g=(rplanet/ro)^alp*(per/po)^bet.
    boolean normalized specifies whether one normalizes the shape function over
    rmin-rmax and pmin-pmax or not
    """

    ro = (rmin+rmax) / 2.0
    po = (pmin+pmax) / 2.0
    rscl = rp / ro
    pscl = per / po
    rbrkscl = rpbrk / ro
    if rp >= rpbrk:
        g = (rbrkscl**(alp1-alp2)) * (rscl**alp2) * (pscl**bet)
    else:
        g = (rscl**alp1)*(pscl**bet)

    if normalized:
        if alp1 == -1:
            intr1 = ro * np.log(rpbrk/rmin)
        else:
            intr1 = (rpbrk**(1.0+alp1) - rmin**(1.0+alp1)) / \
                    (1.0+alp1) / ro**alp1

        if alp2 == -1:
            intr2 = (rpbrk) * (rbrkscl**alp1) * np.log(rmax/rpbrk)
        else:
            intr2 = (rmax**(1.0+alp2)*rpbrk**(-alp2) - rpbrk) / \
                    (1.0+alp2)*rbrkscl**alp1

        if bet == -1:
            intp = po*np.log(pmax/pmin)
        else:
            intp = (pmax**(1.0+bet) - pmin**(1.0+bet)) / \
                   (1.0+bet)/po**bet
    n = (intr1+intr2) * intp
    g = g / n
    return g

# Define the function to do all the double integral work
def integral_function(cit):
        """Use the parameters in the class PLDFParameterStruct member
           to perform integration
        """
        result, err = dblquad(singlepbroker, cit.plow, cit.phgh,
                         lambda x: cit.rlow, lambda x: cit.rhgh, 
                         args = (cit.alp1, cit.alp2, cit.rbrk, 
                           cit.beta, cit.rmin, cit.rmax, cit.pmin,
                           cit.pmax, 1))
        result = result * cit.bigf
        if cit.id > 0 and np.mod(cit.id,1000) == 0:
            print "Iteration: %d Value: %f" % (cit.id, result)

        return result

# Create class that will contain all parameters needed for a single integration
class PLDFParameterStruct:
        def __init__(self):
            self.alp1 = 0.0
            self.alp2 = 0.0
            self.rbrk = 0.0
            self.beta = 0.0
            self.bigf = 0.0
            self.rmin = 0.0
            self.rmax = 0.0
            self.pmin = 0.0
            self.pmax = 0.0
            self.rlow = 0.0
            self.rhgh = 0.0
            self.plow = 0.0
            self.phgh = 0.0
            self.id = 0

def occurpy(rlow, rhgh, plow, phgh, 
            source=1, numberProcessors=1, verbose=False):
    """Calculate a planet occurrence rate from the posterior
       samples of the planet distribution function (PLDF) parameters.
       The PLDF samples are from the paper Burke et al.(2015)
       AUTHOR: Christopher J. Burke
       INPUTS:
       rlow - Planet radius [Rearth] lower limit of integration
              float scalar 0.0<rlow<100.0
       rhgh - Planet radius [Rearth] upper limit of integration
              float scalar 0.0<rlow<rhgh<100.0
       plow - Orbital period [day] lower limit of integration
              float scalar 0.0<plow<1000.0
       phgh - Orbital period [day] upper limit of integration
              float scalar 0.0<plow<phgh<1000.0
       source - Select data source
              integer scalar 1<source<1
              1 - Burke et al.(2015) 0.75<Rp<2.5 Rearth; 50<Porb<300 day
       numberProcessors - Number of parallel processes to employ for the
                          calculation
                          integer scalar 0<numberProcessors<Computers limit
       verbose - Show percentage progress

       OUTPUTS:
       resultoutput - Results of the calculation and posterior statistics
                      It is of type class OccurOutput

       COMMENTS: With a single processor on Intel Xeon 3.5Ghz
                 this can take 5 mins.  With 6 cores it takes 47 secs
    """

    # Check inputs
    if rlow < 0 or rlow > 100:
        raise ValueError("rlow Value %f out of range" % (rlow,))
    if rhgh < 0 or rhgh <= rlow or rhgh > 100:
        raise ValueError("rhgh Value %f out of range or rhgh<rlow"
                         % (rhgh,))
    if plow < 0 or plow > 1000:
        raise ValueError("plow Value %f out of range" % (plow,))
    if phgh < 0 or phgh <= plow or phgh > 1000:
        raise ValueError("phgh Value %f out of range or phgh<plow"
                         % (phgh,))
    if source < 1 or source > 1:
        raise ValueError("source Value %d out of range" % (source,))
    if numberProcessors > multiprocessing.cpu_count():
        raise ValueError("numberProcessors %d out of range max: %d"
                         % (numberProcessors, multiprocessing.cpu_count()))


    # Load in data series from text file
    # This contains hardcoded file names and source limits
    if source == 1:
        alp1, alp2, rbrk, beta, bigf = np.loadtxt(
          "burke_2015_base.txt", comments='#', usecols=(0,1,2,3,4), 
          unpack=True)
        # Also hardcode the limits of the source
        rmin = np.full_like(alp1,0.75)
        rmax = np.full_like(alp1,2.5)
        pmin = np.full_like(alp1,50.0)
        pmax = np.full_like(alp1,300.0)

    # Data is all setup.  Now package the vectors into a list of class
    # Create a list of the PLDFParameterStruct class
    all_iterations = []
    for i in range(alp1.size):
        cit = PLDFParameterStruct()
        cit.alp1 = alp1[i]
        cit.alp2 = alp2[i]
        cit.rbrk = rbrk[i]
        cit.beta = beta[i]
        cit.bigf = bigf[i]
        cit.rmin = rmin[i]
        cit.rmax = rmax[i]
        cit.pmin = pmin[i]
        cit.pmax = pmax[i]
        cit.rlow = rlow
        cit.rhgh = rhgh
        cit.plow = plow
        cit.phgh = phgh
        if verbose:
            cit.id = i
        all_iterations.append(cit)
    # All integral iterations are now loaded in all_iterations

    # Now do all iterations
    pool = multiprocessing.Pool(processes=numberProcessors)
    planoccur = pool.map(integral_function, all_iterations)
    pool.close()
    pool.join()

    # Convert list to numpy array
    planoccur = np.array(planoccur)

    # Create a class that contains the output occurrence rate vector
    # along with many summary statistics for the posterior
    class OccurOutput:
        percentswanted = np.array([0.135, 2.275, 15.865, 50.0,
                                   84.135, 97.725, 99.865])
        limitswanted = np.array([0.27, 4.55, 31.73, 68.27, 95.45, 99.73])
        def __init__(self):
            self.values = []
            self.percentsresults = []
            self.limitsresults = []
            self.median = 0.0
            self.average = 0.0
            self.sigmas = []
            self.radiusextrapolated = False
            self.periodextrapolated = False
    # Fill the output class
    outputresults = OccurOutput()
    outputresults.values = planoccur
    outputresults.average = planoccur.mean()
    outputresults.percentsresults = np.percentile(planoccur, 
                                                  outputresults.percentswanted)
    outputresults.median = outputresults.percentsresults[3]
    outputresults.limitsresults = np.percentile(planoccur, 
                                                outputresults.limitswanted)
    outputresults.sigmas = outputresults.percentsresults - outputresults.median
    # Check for extrapolation involved
    if rlow < rmin[0] or rhgh > rmax[0]:
        outputresults.radiusextrapolated = True
    if plow < pmin[0] or phgh > pmax[0]:
        outputresults.periodextrapolated = True

    return outputresults

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("rlow", type=float,
                        help="Planet radius lower limit")
    parser.add_argument("rhgh", type=float,
                        help="Planet radius upper limit")
    parser.add_argument("plow", type=float,
                        help="Orbital period lower limit")
    parser.add_argument("phgh", type=float,
                        help="Orbital period upper limit")
    parser.add_argument("-src", type=int,
                        help="Data source (integer)", choices=[1], default=1)
    parser.add_argument("-np", type=int,
                        help="Number of processes for calculation",
                        choices=range(1,multiprocessing.cpu_count()),
                        default=1)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show progress")
    args = parser.parse_args()

    # Run the calculation
    outputresults = occurpy(args.rlow, args.rhgh, args.plow, args.phgh,
                            source=args.src, numberProcessors=args.np,
                            verbose=args.verbose)
    # Type out results of calculation
    print "Average Result %f" % outputresults.average
    print "Median Result %f" % outputresults.median
    print "3-sigma low value: %f offset from median: %f" % \
           (outputresults.percentsresults[0], outputresults.sigmas[0])
    print "2-sigma low value: %f offset from median: %f" % \
           (outputresults.percentsresults[1], outputresults.sigmas[1])
    print "1-sigma low value: %f offset from median: %f" % \
           (outputresults.percentsresults[2], outputresults.sigmas[2])
    print "1-sigma high value: %f offset from median: %f" % \
           (outputresults.percentsresults[4], outputresults.sigmas[4])
    print "2-sigma high value: %f offset from median: %f" % \
           (outputresults.percentsresults[5], outputresults.sigmas[5])
    print "3-sigma high value: %f offset from median: %f" % \
           (outputresults.percentsresults[6], outputresults.sigmas[6])
    if outputresults.radiusextrapolated:
        print "EXTRAPOLATING in radius"
    if outputresults.periodextrapolated:
        print "EXTRAPOLATING in period"

