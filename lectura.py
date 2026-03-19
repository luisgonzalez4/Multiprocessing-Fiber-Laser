import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# Leer archivo de espectros
# -----------------------------------
archivo = "/Users/luisgonzalez/Desktop/FLML/Spec.txt"
df = pd.read_csv(archivo,sep="\t",header=None)
S=np.df.values
plt.plot(S[1,:])
plt.show()
