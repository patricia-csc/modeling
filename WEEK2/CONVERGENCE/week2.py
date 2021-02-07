###############################
# POPA-MIHAI PATRICIA         #
# FEBRUARY 2021               #
# PROGRAM FOR MD LAB, WEEK 2  #
###############################

import matplotlib.pyplot as plt
import sys, os
import pandas as pd
import math
import numpy as np

TO_KCAL = 4.184
dir_path = os.path.dirname(os.path.realpath(__file__))

def clu_vs_cutoff(data):
    plt.figure()
    plt.xlabel("Clustering Cut-Off")
    plt.ylabel("Number of Clusters")
    x_val = [x[0] for x in data]
    y_val = [x[1] for x in data]
    plt.plot(x_val, y_val, 'o')
    plt.savefig("clusters_vs_cut-off.png")

def rmsd_lambda(lambdas, data):
    plt.figure()
    for i in range(0, len(data)):
        if lambdas[i] in [0.2, 0.4, 0.6, 0.8, 1.0]:
            rmsd = pd.Series(data[i] * 10)
            rmsd.plot.kde(bw_method=0.1, label=("λ="+str(lambdas[i])))
    plt.xlim(0, 0.04)
    plt.xlabel("RMSD (Å)")
    plt.ylabel("Probability")
    plt.legend(loc="upper right")
    plt.savefig("prob_distrib_lambda.png")

def dhdl_lambda(lambdas, data, filename):
    plt.figure()
    vals = [mean_and_sigma(data[i], lambdas[i])[0] for i in range(0, len(lambdas))]
    integr = [mean_and_sigma(data[i], lambdas[i]) for i in range(0, len(lambdas))]
    print(integr)
    ener = integration(integr, lambdas)
    #print(ener)
    print("Energy of confinement for lambda schedule:", lambdas, "is", 
          ener[0][-1], " ± ", ener[1][-1], "kcal/mol")
    plt.plot(lambdas, vals, 'o')
    plt.xlabel("λ")
    plt.ylabel("Energy (kcal/mol)")
    plt.savefig(filename)

def func(x):
    return x * math.log(x)

def trapezoid_rule(a, b, f, n):
    h = (b - a) / n
    s = f(a) + f(b)
   
    for i in range(1, n):
        s += 2 * f(a + i * h)

    return h / 2 * s

def mean_and_sigma(data, lambda_val):
    for i in range(0, len(data)):
        data[i] = data[i] / TO_KCAL / lambda_val
    mean = sum(data) / len(data)
    sigma = math.sqrt(sum(list(map(lambda n: math.pow((n - mean), 2), data))) /
            (len(data) - 1))
    uncertainty = sigma / math.sqrt(len(data))
    return (mean, uncertainty)

def average(vals, uncs):
    avg = sum(v for v in vals) / len(vals)
    unc = math.sqrt(sum(list(map(lambda n: n ** 2, uncs))))
    return (avg, unc) 

def integration(vals, lambdas):
    s = 0
    summs = []
    for i in range(0, len(vals) - 1):
        a, b = vals[i][0], vals[i + 1][0]
        h = lambdas[i + 1] - lambdas[i]
        tmp = (0.5 * (b + a) * h) 
        s += tmp
        summs.append(round(tmp, 4))
    summs.append(round(s, 4))
    err = [x[1] for x in vals]
    e = 0
    errs = []
    for er in err:
        tmp = er**2
        e += tmp
        tmp = math.sqrt(tmp)
        errs.append(round(tmp, 4))
    e = math.sqrt(e)
    errs.append(round(e, 4))
    return (summs, errs)

def convergence(data, lambdas):
    plt.figure()
    pools = [12.5, 25, 50, 75, 100]
    
    for p in pools:
        pot = []
        for l in range(0, len(lambdas)):
            slc = int(p / 100 * len(data[0][l]))
            temp = [data[i][l][0:slc] for i in range(0, len(data))]
            means = []
            for t in range(0, len(data)): 
                m = mean_and_sigma(temp[t], lambdas[l])
                means.append(m)
            pot.append(means)
        integrations = []
        for i in range(0, len(pot[0])):
            vals = [x[i] for x in pot[0:len(pot)]]
            integrations.append(integration(vals, lambdas))
        averages = []
        for i in range(0, len(integrations[0][0])):
            vals, errs = [], []
            for j in range(0, len(integrations)):
                vals.append(integrations[j][0][i])
                errs.append(integrations[j][1][i])
            averages.append(average(vals, errs))
        vals = [x[0] for x in averages]
        errs = [x[1] for x in averages]
        plt.errorbar(lambdas[:-1], vals[:-1], errs[:-1], marker="o", 
                 linestyle="none", fmt='o', markersize=8, capsize=6,
                 label=str(p)+"%")
        print("For " + str(p) + "%, the energy of confinement is:", 
              str(round(vals[-1], 4)) + " ± " + str(round(errs[-1], 4)) + " kcal/mol")
    plt.legend(loc="best")
    plt.xlabel("λ")
    plt.ylabel("Energy contribution (kcal/mol)")
    plt.savefig("convergence.png")                        

if __name__ == '__main__':
    # CLUSTERS
    clusters = []
    file = open(os.path.join(dir_path + "/SRC" + "/clusters.txt"))
    for line in file:
        try:
            clusters.append((float(line.split(" ")[0]), 
                             float(line.split(" ")[1])))
        except:
            pass
    #clu_vs_cutoff(clusters)
    
    # RMSD vs. LAMBDAS, dH/dl vs. LAMBDAS
    lambdas, rmsd, energies = [], [], []
    #for file in os.listdir(dir_path + "/SRC" + "/RMSDs"):
    for file in os.listdir(dir_path + "/SRC/CONVERGENCE" + "/SET1"):
        lambdas.append(float(file.split(".rms")[0]))
        #file = open(os.path.join("SRC/RMSDs", file))
        file = open(os.path.join("SRC/CONVERGENCE/SET1", file))
        tmp1, tmp2 = [], []
        for line in file:
            try:
                tmp1.append(float(line.split(" ")[2]))
                tmp2.append(float(line.split(" ")[3]))
            except:
                pass
        rmsd.append(tmp1)
        energies.append(tmp2)
    #rmsd_lambda(lambdas, rmsd)
    dhdl_lambda(lambdas, energies, "dhdl_vs_lambda.png")

    # TRAPEZOID RULE
    print("Integral of x * log(x), from 1 to 2:", 
          round(trapezoid_rule(1, 2, func, 100), 4))

    # CONVERGENCE
    lambdas, energies = [], []
    for dirs in os.listdir(dir_path + "/SRC" + "/CONVERGENCE"):
        crr_set = []
        for file in os.listdir(dir_path + "/SRC" + "/CONVERGENCE" + "/" + dirs):
            tmp1 = []
            if file.endswith(".rms"):
                if float(file.split(".rms")[0]) not in lambdas:
                    lambdas.append(float(file.split(".rms")[0]))
                file = open(os.path.join(dir_path + "/SRC" + "/CONVERGENCE" + "/" + dirs, file))
                for line in file:
                    try:
                        tmp1.append(float(line.split(" ")[3]))
                    except:
                        pass
                crr_set.append(tmp1)
        energies.append(crr_set)
    convergence(energies, lambdas)

    # NEW LAMBDA SCHEDULE
    lambdas, energies = [], []
    for file in os.listdir(dir_path + "/SRC" + "/NEW_LAMBDAS"):
        lambdas.append(float(file.split(".rms")[0]))
        file = open(os.path.join("SRC/NEW_LAMBDAS", file))
        tmp = []
        for line in file:
            try:
                tmp.append(float(line.split(" ")[3]))
            except:
                pass
        energies.append(tmp)
    dhdl_lambda(lambdas, energies, "dhdl_vs_lambdas_new.png")
    #plt.show()
