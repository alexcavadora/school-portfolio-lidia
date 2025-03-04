import pandas as pd
import numpy as np
from matplotlib import cbook
import matplotlib.pyplot as plt



broyden = pd.read_csv("IDM_Broyden_DTLZ2_HV.csv")
#print(broyden.head())

gecco = pd.read_csv("IDM_GECCO_DTLZ2_HV.csv")
#print(gecco.head())

micai = pd.read_csv("IDM_MICAI_DTLZ2_HV.csv")
#print(micai.head())

oliver = pd.read_csv("IDM_Oliver_DTLZ2_HV.csv")
#print(oliver.head())


oliver = oliver.drop(oliver.columns[(np.arange(len(oliver.columns)) % 3 != 0)], axis=1)

eval_points = np.arange(50, 5001, 150)


def get_lines(data):
    return data.median().values, data.quantile(0.05).values, data.quantile(0.95).values

def plot_curves(medians, q1, q3, name):
    plt.plot(eval_points, medians, marker='', linestyle='--', label= name + 'Median')
    #plt.plot(eval_points, q1, marker='', linestyle='-', label= name + 'Q1')
    #plt.plot(eval_points, q3, marker='', linestyle='-', label= name + 'Q3')
    plt.fill_between(eval_points,q1,q3, alpha=0.1)
m, q1, q3 = get_lines(broyden)
plot_curves(m, q1, q3, "Broyden")

m, q1, q3 = get_lines(gecco)
plot_curves(m, q1, q3, "Gecco")

m, q1, q3 = get_lines(micai)
plot_curves(m, q1, q3, "Micai")

m, q1, q3 = get_lines(oliver)
plot_curves(m, q1, q3, "Oliver")

plt.legend()
plt.grid(True)
plt.title('Function Evaluation vs HVr\nProblem: DTLZ2')
plt.xlabel("Func. Eval.")
plt.ylabel("HVr")
plt.show()

