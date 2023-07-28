import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.signal import savgol_filter


def detectChangeOfPulse(p,tol=8e-6):
    indicesOfPulse= []
    sizePN = len(p)
    for i in range (sizePN):
        j = i + 1
        if j< sizePN-1:
            if abs(p[i]-p[j])>tol:
                indicesOfPulse.append(j)
    return indicesOfPulse

def expF(x,a,b,c,d):
    # the function that I want is a+b*exp(cx).
    # I add another parameter as a trick to make sure the optimisation converge (give a good result).
    # it's not neccessary.
    return a+b*np.exp(c*x+d)

def calculate_derivative(y, h=1):
    n = len(y)
    derivative = []

    for i in range(n):
        if i == 0:  # Forward difference
            dy = (y[i + 1] - y[i]) / h
        elif i == n - 1:  # Backward difference
            dy = (y[i] - y[i - 1]) / h
        else:  # Average of upper and lower differences
            dy = (y[i + 1] - y[i - 1]) / (2 * h)

        derivative.append(dy)

    return derivative
def isPotentiation(y,dp=1):
    
    # derive the data and get the sign of the derivative if positive potentiation if negative depression.
    #only need a few data point. 
    # because the function need to be strictly monotous, eitheir decreasing or increasing, we should only need 2 points.
    #BUT because of the sinusoidal signal we have to take more.
    dy = calculate_derivative(y,dp)
    return np.mean(dy) > 0
    

def testEq(f,x,y,p=[]):
    m =100000
    popt =[]
    if len(p)!=0:
        popt,_ = curve_fit(f,x,y,p0=p,maxfev=m,method='lm')
    else:
        popt,_ = curve_fit(f,x,y,maxfev=m,method='lm')
    ypred = f(x,*popt)
    r2=r2_score(y,ypred)
    return (ypred,r2,popt)



def get_STDP_param_from_data(dir_path = os.path.expanduser("~/data"),pn='Pulse number', cn= 'Conductance',
            reduceDataSize = 15,filterOn=True,useLinearRegressionMethod= True, useSTDP =True):
    # loop over all files in the directory
    taupreList = []
    taupostList =[]
    ApreList =[]
    ApostList = []
    g_mins = []
    g_maxs =[]
    names= []
    for file_name in os.listdir(dir_path):
        print(f'get parameter of Fit for {file_name}')
        # check if the file is an xls file
        if file_name.endswith('.xls') or file_name.endswith('.xlsx'):
            # read the xls file
            file_path = os.path.join(dir_path, file_name) # get the path of the file

            filename=os.path.splitext(file_name)[0]
            df = pd.read_excel(file_path) # read the file and store it in the dataframe
            x = np.array(df[pn])
            y = np.array(df[cn])
            changesOfPulse = detectChangeOfPulse(y) # get the pulse limit between depreciation and potentiation.
            potdep =  np.split(y, changesOfPulse)
            # set the size
            if reduceDataSize > 0:
                x = x[:reduceDataSize]
                for e in range(len(potdep)):
                    potdep[e] = potdep[e][:reduceDataSize]
            potentiation = isPotentiation(potdep[0])
            linear_regression_gmin = 20e-6
            linear_regression_gmax = 40e-6
            if useSTDP:
                print("you are using the STDP mode")
                print("this means that your data are the stdp")
            if useLinearRegressionMethod:
                print()
                print("You are using the linear regression method as we derive the data first gmin and gmax cannot be determine you have to enter them yourself gmin is the minimum value of the conductance respectively gmax is the maximum")
                print("then change the value of the parameter linear_regression_gmin and linear_regression_gmax to the value you see.")
                print(f"linear_regression_gmin: {linear_regression_gmin} S, and linear_regression_gmax: {linear_regression_gmax} S")
                print()
            else:
                print("you are using the curve fit method for fitting the curve it uses a Levenberg Marquardt algorithm which can be non converging toward a solution")
                print("Make sure you define well the parameter p0 which is the first guess of the algorithm to get an R2 score that is good enough")
            A_post = []
            A_pre = []
            tau_post = []
            tau_pre = []
            g_max = []
            g_min =[]
            if filterOn :
                print('filter is ON you have to set the parameter of filterParameter.')
                print('it will determine how much the data are smoothed.')
                print('the closer filterParameter is to the value of len(potdep[e]) the smoother the curve is')
                for e in range(len(potdep)):
                    filterParameter= int(2*len(potdep[e])/3)
                    potdep[e] = savgol_filter(potdep[e], filterParameter, 2)
            for e in potdep:
                #Fit the data 
                if useSTDP:
                    dy = np.log(np.abs(e))
                    a, b = np.polyfit(x,dy, 1)
                    if potentiation:
                        A_post.append(np.exp(b))
                        tau_post.append(-1/a)
                        g_min.append(linear_regression_gmin)
                    else:
                        A_pre.append(np.exp(b))
                        tau_pre.append(-1/a)
                        g_max.append(linear_regression_gmax)
                else:
                    if useLinearRegressionMethod:
                        dy = np.log(np.abs(calculate_derivative(e)))
                        a, b = np.polyfit(x,dy, 1)
                        if potentiation:
                            A_post.append(np.exp(b))
                            tau_post.append(-1/a)
                            g_min.append(linear_regression_gmin)
                        else:
                            A_pre.append(np.exp(b))
                            tau_pre.append(-1/a)
                            g_max.append(linear_regression_gmax)
                    else: 
                        r2 = 0
                        p0 = [4e-5,(1e-6 if potentiation else -1e-6),-1,100]
                        while r2<0.95:
                            p0 =10*p0
                            _,r2,param =testEq(expF,x,e,p0)
                            print(r2)
                        if potentiation:
                            A_post.append(param[0])
                            tau_post.append(param[1])
                            g_min.append(param[2])
                        else:
                            A_pre.append(param[0])
                            tau_pre.append(param[1])
                            g_max.append(param[2])
                potentiation= not potentiation

            g_min = np.mean(g_min)
            g_max = np.mean(g_max)
            tau_pre = np.mean(tau_pre)
            tau_post = np.mean(tau_post)
            A_pre = np.mean(A_pre)
            A_post = np.mean(A_post)



            g_mins.append(g_min)
            g_maxs.append(g_max)
            taupreList.append(tau_pre)
            taupostList.append(tau_post)
            ApostList.append(A_post)
            ApreList.append(A_pre)
            names.append(filename)         
            print('stdp pre equation: {:.2E} + {:.2E} * exp(-x/{:.2f})'.format(g_min,A_pre,tau_pre))
            print('stdp post equation: {:.2E} {:.2E} * exp(-x/{:.2f})'.format(g_max,A_post,tau_post))
    return {'g_min': g_mins,
             'g_max':g_maxs,
             'tau_pre':taupreList,
             'tau_post':taupostList,
             'A_post':ApostList,
             'A_pre': ApreList,
             'filenames':names}
