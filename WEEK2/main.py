###############################
# POPA-MIHAI PATRICIA         #
# JANUARY 2021                #
# PROGRAM FOR MD LAB, WEEK 2  #
###############################

import matplotlib.pyplot as plt
import sys, os
import pandas as pd
import math
import numpy as np

TO_KCAL = 4.184
dir_path = os.path.dirname(os.path.realpath(__file__))

def mean_and_sigma(data):
    mean = sum(data) / len(data)
    sigma = math.sqrt(sum(list(map(lambda n: math.pow((n - mean), 2), data))) /
            (len(data) - 1))
    uncertainty = sigma / math.sqrt(len(data))
    return (mean, uncertainty)

def clu_vs_cutoff(data):
    plt.figure()
    plt.xlabel("Clustering Cut-Off")
    plt.ylabel("Number of Clusters")
    x_val = [x[0] for x in data]
    y_val = [x[1] for x in data]
    plt.plot(x_val, y_val, 'o')
    plt.savefig("clusters_vs_cut-off.png")

def prob_distrib_lambda(lambdas, data):
    plt.figure()
    for i in range(0, len(data)):
        if lambdas[i] in [0.2, 0.4, 0.6, 0.8, 1.0]:
            rmsd = pd.Series(data[i] * 10)
            #rmsd.plot(kind = "hist", density = True, bins = 100, alpha=0.65)
            rmsd.plot.kde(bw_method=0.1, label=("λ="+str(lambdas[i])))
    plt.xlim(0, 0.04)
    plt.xlabel("RMSD (Å)")
    plt.ylabel("Probability")
    plt.legend(loc="upper right")
    plt.savefig("prob_distrib_lambda.png")

def func(x):
    return x * math.log(x)

def trapezoid_rule(a, b, f, n):
    h = (b - a) / n
    s = f(a) + f(b)
   
    for i in range(1, n):
        s += 2 * f(a + i * h)

    return h / 2 * s

# NOT USED
def _old_integration(pts, data):
    # Makes sure values are sorted
    sorter = tuple(zip(pts, data))
    sorted(sorter)
    pts, data = zip(*sorter)

    s = 0
    vals = [mean_and_sigma(data[i])[0] / pts[i] for i in range(0, len(pts))]
    for i in range(0, len(pts) - 2):
        a, b = vals[i], vals[i + 1]
        h = pts[i + 1] - pts[i]
        s += (0.5 * (b + a) * h)
    return round(s/TO_KCAL, 4)

# Function to do Thermodynamic Integration
def _integration(lambdas, pot):
    # Convert data from kJ to kcal
    vals = [mean_and_sigma(pot[i])[0] / lambdas[i] / TO_KCAL for i in range(0, len(lambdas))]
    err = [mean_and_sigma(pot[i])[1] / lambdas[i] / TO_KCAL for i in range(0, len(lambdas))]
  
    s = 0  
    for i in range(0, len(lambdas) - 2):
        a, b = vals[i], vals[i + 1]
        h = lambdas[i + 1] - lambdas[i]
        s += (0.5 * (b + a) * h)
    
    e = 0
    for er in err:
        e += er**2
    e = math.sqrt(e)
    return (round(s, 2), round(e, 2))

def pot_vs_lambda_og_lambdas_err(lambdas, pot):
    plt.figure()
    vals = [mean_and_sigma(pot[i])[0] / lambdas[i] / TO_KCAL for i in range(0, len(lambdas))]
    err = [mean_and_sigma(pot[i])[1] / lambdas[i] / TO_KCAL for i in range(0, len(lambdas))]
    
    og_lambdas, og_vals, og_err = [], [], []
    for i in range(0, len(lambdas)):
        if lambdas[i] in [0.2, 0.4, 0.6, 0.8, 1.0]:
            og_vals.append(vals[i])
            og_err.append(err[i])
            og_lambdas.append(lambdas[i])

    plt.errorbar(og_lambdas, og_vals, og_err, marker="o", 
                 linestyle="none", fmt='o', markersize=8, capsize=6)
    plt.xlabel("λ")
    plt.ylabel("Energy (kcal/mol)")
    plt.savefig("pot_vs_lambda_OG-lambdas_WITHERR.png")

def pot_vs_lambda_og_lambdas(lambdas, pot):
    plt.figure()
    vals = [mean_and_sigma(pot[i])[0] / lambdas[i] / TO_KCAL for i in range(0, len(lambdas))]

    og_lambdas, og_vals = [], []
    for i in range(0, len(lambdas)):
        if lambdas[i] in [0.2, 0.4, 0.6, 0.8, 1.0]:
            og_vals.append(vals[i])
            og_lambdas.append(lambdas[i])

    plt.plot(og_lambdas, og_vals, 'o')
    plt.xlabel("λ")
    plt.ylabel("Energy (kcal/mol)")
    plt.savefig("pot_vs_lambda_OG-lambdas.png")

def pot_vs_lambda(lambdas, pot):
    plt.figure()
    vals = [mean_and_sigma(pot[i])[0] / lambdas[i] / TO_KCAL for i in range(0, len(lambdas))]

    plt.plot(lambdas, vals, 'o')
    plt.xlabel("λ")
    plt.ylabel("Energy (kcal/mol)")
    plt.savefig("pot_vs_lambda.png")

if __name__ == '__main__':    
    lambdas, clusters, rmsd, pot = [], [], [], []
    
    for file in os.listdir(dir_path + "/SRC"):
        tmp1, tmp2 = [], []        
        if file.endswith(".rms"):
            lambdas.append(float(file.split(".rms")[0]))
            file = open(os.path.join("SRC", file))
            for line in file:
                try:
                    tmp1.append(float(line.split(" ")[2]))
                    tmp2.append(float(line.split(" ")[3]))
                except:
                    pass
            rmsd.append(tmp1)
            pot.append(tmp2)
        
        elif file.endswith(".txt"):
            file = open(os.path.join("SRC", file))
            for line in file:
                try:
                    clusters.append((float(line.split(" ")[0]), 
                                     float(line.split(" ")[1])))
                except:
                    pass

    clu_vs_cutoff(clusters)
    prob_distrib_lambda(lambdas, rmsd)
    pot_vs_lambda(lambdas, pot)
    pot_vs_lambda_og_lambdas(lambdas, pot)
    pot_vs_lambda_og_lambdas_err(lambdas, pot)

    print("Integral of x * log(x), from 1 to 2:", 
          round(trapezoid_rule(1, 2, func, 100), 4))

    conf_energy = _integration(lambdas, pot)
    print("Free Energy of Confinement in Bound State:",
          conf_energy[0], "±", conf_energy[1], "kcal/mol")

    plt.show()
