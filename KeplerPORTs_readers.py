import shelve
import numpy as np
import scipy.io as sio
import KeplerPORTs_utils as kpu

"""Module of functions to read kepler occurrence rate data products
"""


def read_stellar_table(filename,output_prefix=''):
    """Reads in the full Kepler stellar table and saves the 
       information relevant to an occurrence rate calculation
       as a dictionary.  The Kepler stellar table search page is

       http://exoplanetarchive.ipac.caltech.edu/applications/
           TblSearch/tblSearch.html?app=ExoSearch&config=keplerstellar

       The Q1-Q16 table is downloaded from this URL (205Mb)

       http://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/
           nph-nstedAPI?table=q1_q16_stellar&format=ipac&select=*

       The Q1-Q17 table is downloaded from this URL (206Mb)

       http://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/
           nph-nstedAPI?table=q1_q17_dr24_stellar&format=ipac&select=*

       Save the resulting file and use its filename as input to this
       function.
       INPUT:
           filename - Stellar table filename
           output_prefix - output filename for shelve save
                           default is not to save shelve file
       OUTPUT:
           stellar_dict - Dictionary using the KIC ids as the key.
                          The value is an instance of
                            kepler_single_comp_data class defined in
                            KeplerPORTs_utils module
    """
    # Read in file that is fixed width file
             #kepid                                             dens_err2
    delimseq=[11,26,6,10,10,8,10,10,7,9,9,7,10,10,8,12,12,11,11,11,\
             #prov_sec                                          kmag_err
              21,7,7,5,5,71,26,16,11,26,11,26,21,21,21,21,9,9,9,9,9,9,\
             #dutycycle dataspan
              10,11]
    # last 27 columns are 13 wide
    delimseq.extend([17] * 27)
    delimseq.append(18)
    dtypeseq=(int,'S1',int,int,int,float,float,float,float,float,\
              float,float,float,float,float,float,float,float,\
              float,float,'S1',float,int,int,int,'S1','S1','S1',float,\
              'S1',float,'S1','S1','S1','S1','S1',float,float,float,float,\
              float,float,float,float,float,float,float,float,float,float,\
              float,float,float,float,float,float,float,float,float,float,\
              float,float,float,float,float,float,float,float,float,float,\
              float,float)
#    usecolseq=[0,2,5,14,42,43]
#    usecolseq.extend(range(44,72))
    BIG = np.genfromtxt(filename, skip_header=154, delimiter=delimseq,
                        dtype=dtypeseq)
    # Make the list of kepler_single_comp_data class 
    all_datas = []
    all_keys = []
    pulsedurations = np.array([1.5,2.0,2.5,3.0,3.5,4.5,5.0,
                               6.0,7.5,9.0,10.5,12.0,12.5,15.0])
    tmp1 = np.zeros_like(pulsedurations)
    tmp2 = np.zeros_like(pulsedurations)
    for i in range(BIG.size):
        cur = kpu.kepler_single_comp_data()
        cur.id = BIG[i][0]
        cur.rstar = BIG[i][14]
        cur.logg = BIG[i][5]
        cur.teff = BIG[i][2]
        cur.dataspan = BIG[i][43]
        cur.dutycycle = BIG[i][42]
        cur.pulsedurations = np.copy(pulsedurations)
        for j in range(tmp1.size):
          jj = j + 44
          kk = j + 58
          tmp1[j] = BIG[i][jj]
          tmp2[j] = BIG[i][kk]
        cur.cdpps = np.copy(tmp2)
        cur.mesthresh = np.copy(tmp1)
        all_datas.append(cur)
        all_keys.append(cur.id)
    # Now zip it all together into dictionary with kic as key
    stellar_dict = dict(zip(all_keys, all_datas))

    if output_prefix:
        # write out the stellar dictionary with shelve
        newshelf = shelve.open(output_prefix + '.shelf')
        newshelf['stellar_dict'] = stellar_dict
        newshelf.close

    return stellar_dict
