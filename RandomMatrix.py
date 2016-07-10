import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def Wishart(n,p,baseVariance):
    returnVal=np.zeros([n,p])
    for rowIndex in range(0,n):
        nextRow= np.random.multivariate_normal(np.zeros(p),baseVariance*np.identity(p))
        returnVal[rowIndex,:]=nextRow
    return np.dot(np.transpose(returnVal),returnVal)
    
def InverseWishart(n,p,baseVariance):
    baseVariance=1/(baseVariance+.0)
    returnVal=Wishart(n,p,baseVariance)
    return np.linalg.linalg.inv(returnVal)
    
def GaussianUnitaryEnsemble(n,var):
    returnVal= var*np.random.randn(n,n) + var*1j*np.random.randn(n,n)
    returnVal=(np.transpose(np.conj(returnVal))+returnVal)/(2*np.sqrt(n))
    return returnVal
    
def GaussianOrthogonalEnsemble(n,var):
    returnVal= var*np.random.randn(n,n)
    returnVal=(np.transpose(np.conj(returnVal))+returnVal)/(2*np.sqrt(n))
    return returnVal
    
    
#print Wishart(3,2,5)
#print InverseWishart(3,2,5)
#print GaussianUnitaryEnsemble(4,3)
allEigen=[]
for i in range(0,10):
    x=GaussianUnitaryEnsemble(500,1)
    eigen= np.linalg.linalg.eigvals(x)
    eigen=np.round(eigen,10)
    allEigen.extend(eigen)

plt.hist(allEigen)
plt.title("GUE Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")

fig = plt.gcf()
plt.show()
