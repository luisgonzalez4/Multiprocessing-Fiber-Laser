import numpy as np
import matplotlib.pyplot as plt

P=np.linspace(0,15,1000)
T=0.5+(2/5)*np.cos(6*P*(1.3-3))
plt.plot(P,T)
plt.xlabel('Power, W')
plt.ylabel('Transmission')
plt.show()