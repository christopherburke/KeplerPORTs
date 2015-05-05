"""Worked example of the KeplerPORTs_utils module
   to recreate figure 1 of burke et al.(2015).
   This shows how to create a single target Kepler pipeline
   detection grid.  To run at the command line enter
   python test_comp_grid.py
   Output will be  test_comp_grid.eps and .png of the figure
"""
import numpy as np
import matplotlib.pyplot as plt
import KeplerPORTs_utils as kpu

# Instantiate pipeline completeness class structure
doit = kpu.kepler_single_comp_data()
# Test out kepler-22b as in the burke et al.(2015) paper
doit.id = 10593626
doit.period_want = np.linspace(10.0, 700.0, 6000)
doit.rp_want = np.linspace(0.5, 2.5, 3000)
doit.rstar = 0.98
doit.logg = 4.44
doit.deteffver = 2
doit.ecc = 0.0
doit.dataspan = 1426.7
doit.dutycycle = 0.879
doit.pulsedurations = [1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.0, 6.0,
                       7.5, 9.0, 10.5, 12.0, 12.5, 15.0]
doit.cdpps = [36.2, 33.2, 31.0, 29.4, 28.0, 26.1, 25.4, 24.2, 
              23.1, 22.4, 21.9, 21.8, 21.7, 21.5]
doit.mesthresh = np.full_like(doit.pulsedurations,7.1)

x1,x2 = kpu.kepler_single_comp(doit)

X, Y = np.meshgrid(doit.period_want, doit.rp_want)



myblack = tuple(np.array([0.0, 0.0, 0.0]) / 255.0)
mynearblack = tuple(np.array([75.0, 75.0, 75.0]) / 255.0)
myblue = tuple(np.array([0.0, 109.0, 219.0]) / 255.0)
myred = tuple(np.array([146.0, 0.0, 0.0]) / 255.0)
myorange = tuple(np.array([219.0, 209.0, 0.0]) / 255.0)
myskyblue = tuple(np.array([182.0, 219.0, 255.0]) / 255.0)
myyellow = tuple(np.array([255.0, 255.0, 109.0]) / 255.0)
mypink = tuple(np.array([255.0, 182.0, 119.0]) / 255.0)
labelfontsize=15.0
tickfontsize=14.0
datalinewidth=3.0
plotboxlinewidth=3.0

wantFigure = 'test_comp_grid'
plt.figure(figsize=(6,4.5),dpi=300)
ax = plt.gca()
ax.set_position([0.125, 0.125, 0.825, 0.825])
uselevels = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

CS2 = plt.contour(X, Y, x1, levels=uselevels, linewidth=datalinewidth, 
                   colors=(myblue,) * len(uselevels))
plt.clabel(CS2, inline=1, fontsize=labelfontsize, fmt='%1.2f', 
           inline_spacing=10.0, fontweight='ultrabold')
CS1 = plt.contourf(X, Y, x1, levels=uselevels, cmap=plt.cm.bone)    
plt.xlabel('Period [day]', fontsize=labelfontsize, fontweight='heavy')
plt.ylabel('R$_{p}$ [R$_{\oplus}$]', fontsize=labelfontsize, 
            fontweight='heavy')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(plotboxlinewidth)
    ax.spines[axis].set_color(mynearblack)
ax.tick_params('both', labelsize=tickfontsize, width=plotboxlinewidth, 
               color=mynearblack, length=plotboxlinewidth*3)
plt.savefig(wantFigure+'.png',bbox_inches='tight')
plt.savefig(wantFigure+'.eps',bbox_inches='tight')
plt.show()

