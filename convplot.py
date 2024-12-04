import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
x=pd.read_csv("plot.csv")
x=np.array(x)
y=np.arange(100)
fig,ax=plt.subplots()
ax.plot(y,x[3])
ax.plot(y,x[0])
ax.plot(y,x[1])
ax.plot(y,x[2])
ax.legend(('PSO','PSOGA','PSODWCN','PSODWCNGA'))
plt.savefig("dev.png")