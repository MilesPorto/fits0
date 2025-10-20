import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

xmin=1.0
xmax=20.0
npoints=12
sigma=0.2
lx=np.zeros(npoints)
ly=np.zeros(npoints)
ley=np.zeros(npoints)
pars=[0.5,1.3,0.5]

from math import log
def f(x,par):
    return par[0]+par[1]*log(x)+par[2]*log(x)*log(x)

from random import gauss
def getX(x):  # x = array-like
    step=(xmax-xmin)/npoints
    for i in range(npoints):
        x[i]=xmin+i*step
        
def getY(x,y,ey):  # x,y,ey = array-like
    for i in range(npoints):
        y[i]=f(x[i],pars)+gauss(0,sigma)
        ey[i]=sigma

# get a random sampling of the (x,y) data points, rerun to generate different data sets for the plot below
getX(lx)
getY(lx,ly,ley)

fig, ax = plt.subplots()
ax.errorbar(lx, ly, yerr=ley)
ax.set_title("Pseudoexperiment")
fig.show()


# *** modify and add your code here ***
nexperiments = 1000  # for example
par_a = np.zeros(nexperiments)
par_b = np.zeros(nexperiments)
par_c = np.zeros(nexperiments)
chi2 = np.zeros(nexperiments)
chi2_reduced = np.zeros(nexperiments)

# perform many least squares fits on different pseudo experiments here
# fill histograms w/ required data
for i in range(0,nexperiments):
  getX(lx)
  getY(lx,ly,ley)
  A = np.zeros((npoints, 3))
  A[:,0] = 1.0
  A[:,1] = np.log(lx)
  A[:,2] = np.log(lx)**2

  Aw = A / ley[:, None]
  yw = (ly / ley).reshape(npoints,1)

  theta = inv(Aw.T @ Aw) @ (Aw.T @ yw)

  a,b,c = theta.flatten()
  chi2_sample = np.sum(((ly - (a + b*np.log(lx) + c*np.log(lx)**2)) / ley)**2)
  dof = npoints - 3
  reduced_chi2 = chi2_sample / dof
  par_a[i] = a
  par_b[i] = b
  par_c[i] = c
  chi2_reduced[i] = reduced_chi2
  chi2[i]=chi2_sample

mean_chi2 = np.mean(chi2)
std_chi2 = np.std(chi2, ddof=1)
e_mean = npoints-3
e_std = np.sqrt(2*(npoints-3))
print(mean_chi2, std_chi2, e_mean, e_std)

fig2, axs2 = plt.subplots(2, 2)
plt.tight_layout()
fig, axs = plt.subplots(2, 2)
plt.tight_layout()


# careful, the automated binning may not be optimal for displaying your results!
axs[0, 0].hist2d(par_a, par_b)
axs[0, 0].set_title('Parameter b vs a')

axs[0, 1].hist2d(par_a, par_c)
axs[0, 1].set_title('Parameter c vs a')

axs[1, 0].hist2d(par_b, par_c)
axs[1, 0].set_title('Parameter c vs b')

axs[1, 1].hist(chi2_reduced)
axs[1, 1].set_title('Reduce chi^2 distribution')

axs2[0, 0].hist(par_a)
axs2[0, 0].set_title('Parameter a distribution')

axs2[0, 1].hist(par_b)
axs2[0, 1].set_title('Parameter b distribution')

axs2[1, 0].hist(par_c)
axs2[1, 0].set_title('Parameter c distribution')
#axs2[1, 0].locator_params(axis='x', nbins=5)

axs2[1, 1].hist(chi2)
axs2[1, 1].set_title('chi^2 distribution')

fig2.show()
fig.show()


# **************************************
fig.savefig("fit_correlations.png", dpi=300, bbox_inches='tight')
fig2.savefig("fit_distributions.png", dpi=300, bbox_inches='tight')

pdf_filename = "LSQFit.pdf"
doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
styles = getSampleStyleSheet()

content = []

content.append(Paragraph("Fit Results for 12 points, 0.2 sigma", styles['Title']))
content.append(Spacer(1, 12))

# Add the first figure (istributions)
content.append(Image("fit_distributions.png", width=450, height=350))
content.append(Spacer(1, 24))

summary_text = """
As the number of points increases, the average values of the parameters remain the same, but their distributions become tighter: the standard deviation decreases. Chi^2 also increases which is to be expected, and so does its standard deviation. As the size of the uncertainties increases, the average values of the parameters remain the same, but the standard deviation of the parameters gets bigger. The average value of chi^2 and its standard deviation also remain the same.

The expected values for chi^2 and the standard deviation of chi^2 agree very closely with the experimental values. For example: experimental mean of chi^2: 8.968 
expected chi^2: 9 
experimental stdev: 4.369 
expected stdev: 4.243
"""
content.append(Paragraph(summary_text, styles['Normal']))
content.append(Spacer(1, 24))
# Add the second figure (correl)
content.append(Image("fit_correlations.png", width=450, height=350))


# Add conclusion or comments


doc.build(content)

print(f"PDF saved as {pdf_filename}")

input("hit Enter to exit")
