###############################
# POPA-MIHAI PATRICIA         #
# FEBRUARY 2021               #
# PROGRAM FOR MD LAB, WEEK 3  #
###############################

import matplotlib.pyplot as plt
import sys, os
import pandas as pd
import math
import numpy as np
import colorsys 

# CONSTANTS
toKcal = 4.184
# CURRENT DIRECTORY PATH
dir_path = os.path.dirname(os.path.realpath(__file__))
# COLORS FOR PLOTS
# TO PICK YOUR COLOR, SEARCH RGB TO HLS CONVERTER
# ORDER: HUE, SATURATION, LIGHTNESS
COLORS = [(214, 49, 85),
          (213, 52, 79),
          (212, 55, 71),
          (209, 59, 60),
          (208, 48, 55),
          (209, 41, 49),
          (209, 41, 42)]

# Probability distribution plotter
def prob_distrib(lambdas, CV, x_label, filename):
    plt.figure()
    for i in range(0, len(lambdas)):
        pb_distrib = pd.Series(CV[i])
        pb_distrib.plot.kde(bw_method=0.5, label=('λ='+str(lambdas[i])),
            color=colorsys.hls_to_rgb(COLORS[i][0]/360, COLORS[i][2]/100, COLORS[i][1]/100))
    plt.ylabel('Probability')
    plt.xlabel(x_label)
    plt.legend(loc="upper right")
    plt.savefig(filename)

# Mean and uncertainty
def mean(data):
    mean, mnsq, n = 0, 0, len(data)
    for i in range(0, n):
        mean += data[i]
        mnsq += data[i] * data[i]
    mean /= n
    mnsq /= n
    varia = mnsq - (mean * mean)
    return (mean, varia)  

# Ensamble-averaged dU/dl for given lambda and energy
def dUdl(lambda_val, energies):
    avg = mean(energies)
    return (avg[0] / lambda_val / toKcal,
            math.sqrt(avg[1] / lambda_val / toKcal))

# Ensamble-averaged dU/dl for whole data set
def anal_vba(lambdas, data):
    results = []
    for l in range(0, len(lambdas)):
        temp = []
        for i in range(0, len(data)):
            temp.append(dUdl(lambdas[l], data[i][l]))
        results.append(temp)
    return results

# Integration
def integration(lambdas, data):
    sum, err = 0, 0
    partial_sum, partial_errs = [], []
    dhdl = anal_vba(lambdas, data)
    # loop over CVs
    for i in range(0, len(data[0]) - 1):
        temp_sum, temp_err = 0, 0
        # loop over lambdas
        for j in range(0, len(lambdas) - 1):
            bb = dhdl[j][i][0]
            BB = dhdl[j + 1][i][0]
            hh = lambdas[j + 1] - lambdas[j]
            dA = (BB + bb) * hh / 2
            temp_sum += dA
            temp_err += (
                (dhdl[j][i][1] * dhdl[j][i][1] + 
                dhdl[j + 1][i][1] * dhdl[j + 1][i][1])
                / 4 * hh * hh
            )
        sum += temp_sum
        err += temp_err
        partial_sum.append(temp_sum)
        partial_errs.append(math.sqrt(temp_err))
    return (list(zip(partial_sum, partial_errs)), (sum, math.sqrt(err)))

# Reading data from files
def input_reader(path, *args):
    slice = 0
    if len(args) != 0:
        slice = args[0] / 100 * (args[1] - 8)
    r, theta, phi, THETA, PHI, PSI = [], [], [], [], [], []
    e_r, e_theta, e_phi, e_THETA, e_PHI, e_PSI = [], [], [], [], [], []
    time, lambdas = [], []
    for filename in os.listdir(dir_path + path):
        t1, t2, t3, t4, t5, t6 = [], [], [], [], [], []  
        e_t1, e_t2, e_t3, e_t4, e_t5, e_t6 = [], [], [], [], [], []        
        if filename.endswith(".vba"):
            # Get lambda from filename
            lambdas.append(float(filename.split(".vba")[0]))
            file = open(os.path.join(path[1:], filename))
            for i, line in enumerate(file):
                if i < slice or slice == 0:
                    try:
                        # Get the time only once
                        if filename.startswith("0.1"):
                            time.append(float(line.split(" ")[1]))
                        # Get the CV values
                        t1.append(float(line.split(" ")[2]))
                        t2.append(float(line.split(" ")[4]))
                        t3.append(float(line.split(" ")[6]))
                        t4.append(float(line.split(" ")[8]))
                        t5.append(float(line.split(" ")[10]))
                        t6.append(float(line.split(" ")[12]))
                        e_t1.append(float(line.split(" ")[3]))
                        e_t2.append(float(line.split(" ")[5]))
                        e_t3.append(float(line.split(" ")[7]))
                        e_t4.append(float(line.split(" ")[9]))
                        e_t5.append(float(line.split(" ")[11]))
                        e_t6.append(float(line.split(" ")[13]))
                    except:
                        pass
                else:
                    break
            r.append(t1)
            theta.append(t2)
            phi.append(t3)
            THETA.append(t4)
            PHI.append(t5)
            PSI.append(t6)
            e_r.append(e_t1)
            e_theta.append(e_t2)
            e_phi.append(e_t3)
            e_THETA.append(e_t4)
            e_PHI.append(e_t5)
            e_PSI.append(e_t6)
            file.close()

    return (lambdas, [r, theta, phi, THETA, PHI, PSI], 
            [e_r, e_theta, e_phi, e_THETA, e_PHI, e_PSI])

def convergence(path, percentages):
    energies = []
    num_lines = sum(1 for line in open(dir_path + path + '/0.1.vba'))
    plot_names = ['conv_r.png', 'conv_theta.png', 'conv_phi.png', 
                  'conv_capital_THETA.png', 'conv_capital_PHI.png', 'conv_capital_PSI.png']
    for p in percentages:
        (lambdas, _, data) = input_reader(path, p, num_lines)
        pretty_print(integration(lambdas, data))
        energies.append(anal_vba(lambdas, data))
    # Loop over CVs
    for type in range(0, len(energies[0][0])):
        plt.figure()
        # Loop over simulation times
        for j in range(0, len(energies)):
            energ = []
            # Loop over lambdas
            for i in range(0, len(energies[0])):
                energ.append(energies[j][i][type][0])
            # Plot it
            plt.loglog(lambdas, energ, 'o', label=(str(percentages[j])+"%"))
        plt.xlabel("λ")
        plt.ylabel("Energy (kcal/mol)")
        plt.legend(loc="upper right")
        plt.savefig(plot_names[type])

def pretty_print(data):
    print("Contribution from individual CVs:")
    for (val, err) in data[0]:
        print(str(format(round(val, 5), '.5f')) + " ± " + 
              str(format(round(err, 5), '.5f')) + " kcal/mol")
    print("VBA Free Energy: " + str(format(round(data[1][0], 5), '.5f')) 
          + " ± " + str(format(round(data[1][1], 5), '.5f')) + " kcal/mol")

if __name__ == '__main__':    
    (lambdas, vals, energies) = input_reader('/SRC')
    pretty_print(integration(lambdas, energies))
    prob_distrib(lambdas, vals[0], "r (nm)", "r.png")
    prob_distrib(lambdas, vals[1], "θ", "theta.png")
    prob_distrib(lambdas, vals[2], "ϕ", "phi.png")
    prob_distrib(lambdas, vals[3], "Θ", "capital_THETA.png")
    prob_distrib(lambdas, vals[4], "Φ", "capital_PHI.png")
    prob_distrib(lambdas, vals[5], "Ψ", "capital_PSI.png")

    if os.path.isdir(dir_path + '/SRC/DOUBLE'):
        (lambdas, _, data) = input_reader('/SRC/DOUBLE')
        pretty_print(integration(lambdas, data))

    if os.path.isdir(dir_path + '/SRC/CONVERGENCE'):
        convergence('/SRC/CONVERGENCE', [12.5, 25, 50, 75, 100])
        (lambdas, _, data) = input_reader('/SRC/CONVERGENCE')
        pretty_print(integration(lambdas, data))