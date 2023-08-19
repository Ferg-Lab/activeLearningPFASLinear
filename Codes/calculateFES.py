import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import copy
from scipy.integrate import simps
import math

import rdkit
from rdkit import Chem
from rdkit.Chem.Fragments import *


#### PMF along a CV in a given time block.
def get_block_fes_using_cv_weights(temperature,  simDir, systemDir, colvarFileName, colvarFileContentList, cvmin, cvmax, nbins, cvname, cvbiasname, shiftCV, startTimeInPS, endTimeInPS):
    
    """
    CALCULATE PMF along a given CV using umbrella sampling type reweighting method by reading specific time range in a COLVAR file to obtain free energies or binding constants using block averaging method.
    
    Parameters
    ----------

    
    temperature: Float
            Temperature at which PBMetaD simulations are performed.
    
    simDir: String
        Path to datafiles.
    
    systemDir: String
        Folder containing COLVAR files from PBMetaD simulations.
    
    colvarFileName: String
        Name of the COLVAR file containing CV values and converged BIAS.
    
    colvarFileContentList: List
        List of strings to identify each column in colvarFileName.
    
    cvmin: Float
        Minimum value for CV to calculate histogram for PMF calculations.
    
    cvmax: Float
        Maximum value for CV to calculate histogram for PMF calculations.
    
    nbins: Int
        Number of bins to calculate histogram.
    
    cvname: String
        Target cv to use for obtaining PMF.
    
    cvbiasname: String
        Name of the BIAS column.
    
    shiftCV: Float
        CV value at which PMF is set to zero.
    
    startTimeInPS: Float
        start time in ps to read the COLVAR FILE.
    
    endTimeInPS: Float
        end time in ps to read the COLVAR FILE.
    
    Returns
    -------
    
        feDf: DataFrame
            PMF along CV
    """

    # Boltzmann Constant or R
    kb = 8.31446261815324*0.001 #kJ/molK
    kbt = kb*temperature  #kJ/mol
    
    preFactor = 2*kbt #kJ/mol for entropy correction

    colvarFile = simDir + systemDir + colvarFileName

    # read colvarData into pandas DataFrame
    colvarData = pd.read_csv(colvarFile, delim_whitespace=True, names=colvarFileContentList, comment="#")
        
    # discard initial few frames [frames where free energy is still being mapped]
    colvarData = colvarData [ (colvarData.time > startTimeInPS) & (colvarData.time <= endTimeInPS) ]
    
    # calculate weights of each frame in colvarData
    weights = np.exp( (colvarData[cvbiasname]) /kbt)

    # get unbiased probabilities using umbrella sampling type reweighting approach
    fe,dzfe = np.histogram(colvarData[cvname],range=[cvmin,cvmax],bins=nbins,weights=weights,density=True)
    dzfe = dzfe[:-1]
    
    # combine unbiased probabilities and bins and convert into a pandas DataFrame
    feDf = pd.DataFrame(np.array([dzfe, -kbt*np.log(fe)]).transpose(), columns=[cvname, "fe"])
    feDf.fe += preFactor*np.log(feDf[cvname])
    
    # shift fes to set non-interacting state to zero; consider that point in CV to be equal to shiftCV
    shiftFE = feDf [ (feDf[cvname] <= shiftCV) & (feDf[cvname] >= shiftCV-0.01) ].fe.max()

    # not used here; uncomment if you want to align PMFs according to shiftCV
    #feDf.fe -= shiftFE
    
    return feDf


#### Calculate average PMF profile.
def get_fes_avg_sd(fesDFList, cvname, nbins):
    
    """
    Align PMF profiles using least squares alignment procedure and then 
    calculate average and standard error on the PMF using the block averaging procedure.
    
    Parameters
    ----------
    
    fesDFList: List[DataFrame]
        List of dataframes containing PMF profiles along a CV.
        
    cvname: String
        Target CV along which PMF is obtained.
        
    nbins: Int
        Number of bins used for PMF calculation to obtain average PMF.
        
    Returns
    -------
    
    avgFE: Array
        Block averaged PMF profile.
    
    np.sqrt(varFE): Array
        Standard error of PMF profile.
    
    fesDFListUpdated: List[DataFrame]
        List of updated dataframes containing aligned PMF profiles.
        
    fe_minList: List[Float]
        Global minimum in each PMF profile provided.
        (Not used).
   
    """
    
    localfesDFList = copy.deepcopy(fesDFList)
    
    avgFE = np.zeros(nbins)
    varFE = np.zeros(nbins)
    
    nsamples = len(localfesDFList)
    
    print("nsamples: ", nsamples)
    
    for run in range(nsamples):
        
        avgFE += localfesDFList[run].fe.values
        
    avgFE /= nsamples
    
    # Aligning PMFs by minimizing the differences along CV between PMF of each run and
    # average PMF of all runs. This gives the best alignment.
    # \sum_{i=1 to N} ( (y_i + a) - y_avg_i)^2 is minimum when a = mean(y_avg_i - y_i)
    fesDFListUpdated = []
    for run in range(nsamples):

        # calculate difference between y_avg_i and y_i
        meanPoint = avgFE - localfesDFList[run].fe.values

        meanPoint = meanPoint[~np.isnan(meanPoint)]
        meanPoint = meanPoint[~np.isinf(meanPoint)]
        
        # calculate a
        meanPOint = np.mean(meanPoint)
        
        # align each PMF using a
        temp = localfesDFList[run].fe.values + np.mean(meanPoint)
        
        # update dataframe
        localfesDFList[run].fe = temp
        
        fesDFListUpdated.append(localfesDFList[run])
        
    # Obtain average and error on PMF profiles.
    avgFE = np.zeros(nbins)
    for run in range(nsamples):
        
        avgFE += localfesDFList[run].fe.values
        varFE += localfesDFList[run].fe.values*localfesDFList[run].fe.values
        
        
    avgFE /= nsamples
    varFE /= nsamples
    varFE = (varFE - avgFE*avgFE)/(nsamples/(nsamples - 1))
    
    # shift PMFs and avg PMF by finding the CV that is in the non-interacting state.
    # Here, we use max of CV and max of CV - 1 nm (arbitrary) as a range for finding the cv
    # to shift the PMFs, which here is the minimum free energy in that range.

    #fe_min = 1000
    # any CV is fine because it is the same in all runs.
    #fe_min = find_min_avg(localfesDFList[0][cvname].values, avgFE, fe_min)
    
    fe_minList = []
    for run in range(nsamples):
        fe_min = 1000
        fe_minList.append(find_min_avg(localfesDFList[run][cvname].values, localfesDFList[run].fe.values, fe_min))
        
    
    #return avgFE, np.sqrt(varFE), fesDFListUpdated, fe_min
    return avgFE, np.sqrt(varFE), fesDFListUpdated, fe_minList


#### Helper function to calculate minimum PMF in a given range of CV.
def find_min_avg(cv, fe, fe_min):
    """
    Calculate minimum in the PMF profile in a given CV range.
    
    Parameters
    ----------
    
    cv: String
        CV name
        
    fe: Array
        PMF profile along cv.
        
    fe_min: Float
        Initial guess value to use for calculating minimum.
    
    Returns
    -------
    
    fe_min: Float
        Minimum along PMF profile.
    
    """
    
    for i in range(0, len(cv)):
        if cv[i] > np.max(cv) - 1  and cv[i] < np.max(cv):
            if fe[i] < fe_min:
                fe_min = fe[i]

    return fe_min


# Defining Delta FE as area under the unbiased probability curve obtained from the aligned and shifted free energy profiles
def get_delta_fe_areaMethod(systemFEListUpdated, kT, cvname, cutoff):
    
    """
    Calculate block average binding free energy by computing binding free energy in each
    aligned PMF profile using area under the unbiased probability method.
    
    Parameters
    ----------
    
    systemFEListUpdated: List[DataFrame]
        List of dataframes containing aligned PMF profiles.
        
    kT: Float
        kT in kJ/mol
    
    cvname: String
        CV name
        
    cutoff: Float
        Bound area cutoff.
  
    Returns
    -------
    
    avgFreeEnergy: Float
        Block averaged binding free energy.
    
    sdFreeEnergy: Float
        Standard error on binding free energy.
     
    """
    
    localfesDFListUpdated = copy.deepcopy(systemFEListUpdated)
    
    #avgFE = np.zeros(nbins)
    #varFE = np.zeros(nbins)
    
    nsamples = len(localfesDFListUpdated)
    
    #print("nsamples: ", nsamples)
    
    freeEnergyList = []
    fe_minList = []
    for block in range(nsamples):
        
        boundReg = localfesDFListUpdated[block][ (localfesDFListUpdated[block][cvname] <= cutoff) ]
        boundReg = boundReg[~np.isinf(boundReg.fe)]
        
        fe_min = 1000
        # NOTE: any CV is fine because it is the same in all runs.
        fe_min = find_min_avg(localfesDFListUpdated[block][cvname].values, localfesDFListUpdated[block].fe.values, fe_min)
        
        boundReg.fe = boundReg.fe - fe_min
        
        #print(boundReg)
        #print(kT)
        #print(simps( np.exp(- (boundReg.fe )/(kT)),  boundReg[cvname] ))
        
        freeBoundReg = -(kT)*np.log( simps( np.exp(- (boundReg.fe )/(kT)),  boundReg[cvname] ) )
    
        freeEnergyList.append(freeBoundReg)
        
        fe_minList.append(fe_min)
        
    freeEnergyArray = np.array(freeEnergyList)
    
    avgFreeEnergy = np.mean(freeEnergyArray)
    sdFreeEnergy = np.std(freeEnergyArray)
                  
    return avgFreeEnergy, sdFreeEnergy


# Defining Delta FE as area under the unbiased probability curve obtained from the aligned and shifted free energy profiles
def get_Kb_areaMethod(systemFEListUpdated, kT, cvname, cutoff):
    
    """
    Calculate block average binding constant by computing binding constant from each
    aligned PMF profile using area (spherical integral) under the unbiased probability method.
    
    Parameters
    ----------
    
    systemFEListUpdated: List[DataFrame]
        List of dataframes containing aligned PMF profiles.
        
    kT: Float
        kT in kJ/mol
    
    cvname: String
        CV name
        
    cutoff: Float
        Bound area cutoff.
  
    Returns
    -------
    
    avgFreeEnergy: Float
        Block averaged binding constant.
    
    sdFreeEnergy: Float
        Standard error on binding constant.
     
    """
    
    localfesDFListUpdated = copy.deepcopy(systemFEListUpdated)
    
    #avgFE = np.zeros(nbins)
    #varFE = np.zeros(nbins)
    
    nsamples = len(localfesDFListUpdated)
    
    print("nsamples: ", nsamples)
    
    freeEnergyList = []
    fe_minList = []
    for block in range(nsamples):
        
        boundReg = localfesDFListUpdated[block][ (localfesDFListUpdated[block][cvname] <= cutoff) ]
        boundReg = boundReg[~np.isinf(boundReg.fe)]
        
        fe_min = 1000
        # NOTE: any CV is fine because it is the same in all runs.
        fe_min = find_min_avg(localfesDFListUpdated[block][cvname].values, localfesDFListUpdated[block].fe.values, fe_min)
        
        #print(boundReg)
        
        boundReg.fe = boundReg.fe - fe_min
        
        C0 = 1/(1661*1e-3) # Standard state conc.
        
        freeBoundReg = C0* simps( 4*np.pi*boundReg[cvname]*boundReg[cvname]*np.exp(- (boundReg.fe )/(kT)),  boundReg[cvname] ) 
    
        freeEnergyList.append(freeBoundReg)
        
        fe_minList.append(fe_min)
        
    freeEnergyArray = np.array(freeEnergyList)
    
    avgFreeEnergy = np.mean(freeEnergyArray)
    sdFreeEnergy = np.std(freeEnergyArray)
    
    return avgFreeEnergy, sdFreeEnergy


def write_fes_tofile(method, fileName, analyte, probeSMILES, minFreeEnergy, minSD, formula):
    
    """
    Write binding free energy or binding constant along with PROBE details to a text file (tab separated).
    
    Parameters
    ----------
    
    method: String
        Enhanced sampling method.
    
    fileName: String
        Output file name.
        
    analyte: String
        PFAS/Interferent name.
        
    probeSMILES: String
        Probe smiles string.
        
    minFreeEnergy: Float
        Binding free energy or binding constant.
        
    minSD: Float
        Standard error
        
    formula: String
        Generate some interpretable formulae of probe by counting halogens/carbons/nitrogens.
        
    Returns:
    --------
    
    NONE
        Output file contains the following columns.
            1. Method name
            2. Analyte name
            3. Number of carbon atoms in the probe.
            4. Number of flourine atoms in the probe.
            5. Number of hydrogen atoms in the probe.
            6. Number of hydrogen atoms in the head group of the probe.
            7. Number of carbon atoms in the head group of the probe.
            8. Approximate head group name.
            9. Average binding free energy or binding constant.
            10.Standard error on binding free energy or binding constant.
            11. SMILES string.
            12. Approximate formulae.
    
    """
    
    labelSMILES = Chem.MolFromSmiles(probeSMILES, sanitize=True)
    
    lchain = probeSMILES.count('C')
    
    lfl = probeSMILES.count('F')
    
    lCl = probeSMILES.count('Cl')
    
    lBr = probeSMILES.count('Br')
    
    #print("\nlfl")
    #print(lfl)
    
    X = 'F'
    
    if lCl != 0:
        X = 'Cl'
        lfl = lCl
        lchain -= lCl

    if lBr != 0:
        X = 'Br'
        
    lhydrogen = 2*lchain + 2 - lfl
    
    countP = -1
    if probeSMILES.count('P') > 0:
        
        countP = probeSMILES.count('P')
        probeSMILES = probeSMILES.replace('P', 'N')
        
    rdkProbe = Chem.MolFromSmiles(probeSMILES, sanitize=True)
    
    label_i_head_group_h_count = -1
    label_i_head_group_c_count = -1
    if fr_NH2(rdkProbe) == 1 and fr_quatN(rdkProbe) == 0 :
        label_i_head_group_h_count = 2
        label_i_head_group_c_count = 0
        #print(probes[i], "primary amine")
    elif fr_NH1(rdkProbe) == 1 and fr_quatN(rdkProbe) == 0:
        label_i_head_group_h_count = 1
        label_i_head_group_c_count = 1
        #print(probes[i], "secondary amine")
    elif fr_NH0(rdkProbe) == 1 and fr_quatN(rdkProbe) == 0:
        label_i_head_group_h_count = 0
        label_i_head_group_c_count = 2
        #label_i_head_group = "0"
        #print(probes[i], "tertiary amine")
    elif fr_NH0(rdkProbe) == 1 and fr_quatN(rdkProbe) == 1:
        label_i_head_group_h_count = 0
        label_i_head_group_c_count = 3
        #label_i_head_group = ""
        #print(probes[i], "quaternary amine")
    elif fr_NH2(rdkProbe) == 1 and fr_quatN(rdkProbe) == 1 :
        label_i_head_group_h_count = 2
        label_i_head_group_c_count = 1
        #print(probes[i], "secondary amine cation") 
    elif fr_NH1(rdkProbe) == 1 and fr_quatN(rdkProbe) == 1:
        label_i_head_group_h_count = 1
        label_i_head_group_c_count = 2
        #print(probes[i], "tertiary amine cation")
    elif fr_NH0(rdkProbe) == 0 and fr_quatN(rdkProbe) == 1:
        label_i_head_group_h_count = 3
        label_i_head_group_c_count = 0
        #print(probes[i], "primary amine cation")
    
    
    if label_i_head_group_c_count > 0:
        lchain = lchain - label_i_head_group_c_count
        
        
    if countP == -1:
    
        if label_i_head_group_h_count == -1 and label_i_head_group_c_count == -1:
            label_i = 'C' + str(lchain) + X + str(lfl)
        elif label_i_head_group_h_count == 0:
            label_i = 'C' + str(lchain) + X + str(lfl) + '/N' + '(CH3)' + str(label_i_head_group_c_count)
        elif label_i_head_group_c_count == 0:
            label_i = 'C' + str(lchain) + X + str(lfl) + '/NH' + str(label_i_head_group_h_count)
        elif label_i_head_group_h_count > 0 and label_i_head_group_c_count > 0:
            label_i = 'C' + str(lchain) + X + str(lfl) + '/NH' + str(label_i_head_group_h_count) + '(CH3)' + str(label_i_head_group_c_count)
    else:
        
        if label_i_head_group_h_count == -1 and label_i_head_group_c_count == -1:
            label_i = 'C' + str(lchain) + X + str(lfl)
        elif label_i_head_group_h_count == 0:
            label_i = 'C' + str(lchain) + X + str(lfl) + '/P' + '(CH3)' + str(label_i_head_group_c_count)
        elif label_i_head_group_c_count == 0:
            label_i = 'C' + str(lchain) + X + str(lfl) + '/PH' + str(label_i_head_group_h_count)
        elif label_i_head_group_h_count > 0 and label_i_head_group_c_count > 0:
            label_i = 'C' + str(lchain) + X + str(lfl) + '/PH' + str(label_i_head_group_h_count) + '(CH3)' + str(label_i_head_group_c_count)

            
    with open(fileName, 'a') as deltaGFileObject:
        
        deltaGFileObject.write(method.upper() + '\t' + analyte.upper()  + '\t' + str(lchain) + '\t' + str(lfl) + '\t' + str(lhydrogen) + '\t' + str(label_i_head_group_h_count) + '\t' + str(label_i_head_group_c_count) + '\t' + label_i + '\t' + '\t' + str(minFreeEnergy) + '\t' + str(minSD) + '\t' + Chem.MolToSmiles(labelSMILES) + '\t' + formula + '\n')


        
################################################################################################################
################ 
################ UNUSED METHODS ARE BELOW. 
################ Examples include average potential energy or average structure properties using reweighting.
################ 
################################################################################################################
        
# UNUSED METHOD: Minimum along PMF as binding free energy.
# Defining Delta FE as minimum free energy in the aligned and shifted free energy profiles
def get_delta_fe(avgFEDF, cvname):
    
    """
    Calculate binding free energy by simply taking the minimum of PMF.
    
    Parameters
    ----------
    
    avgFEDF: DataFrame
        Average PMF profile.
        
    cvname: String
        Target CV name.
        
    Returns
    -------
    
    minLocation: Float
        Location of minimum along CV.
        
    minFreeEnergy: Float
        Minimum PMF or binding free energy.
        
    minSD: Float
        Error at minimum PMF.
        
    freeBoundRegArea: Float
        Free energy estimation by taking area under the unbiased probability curve upto 1 nm (bound region)
        from the average PMF profile.
    
    """
    
    
    minLocation = avgFEDF[ ( avgFEDF.avg  == avgFEDF.avg.min() ) ][cvname].values[0]
    minFreeEnergy = avgFEDF[ ( avgFEDF.avg == avgFEDF.avg.min() ) ].avg.values[0]
    minSD = avgFEDF[ ( avgFEDF.avg == avgFEDF.avg.min() ) ].sd.values[0]
    

    #print("average")
    
    boundReg = avgFEDF[ (avgFEDF[cvname] <= 1) ]
    boundReg = boundReg[~np.isinf(boundReg.avg)]
        
    fe_min = 1000
    # any CV is fine because it is the same in all runs.
    fe_min = find_min_avg(avgFEDF[cvname].values, avgFEDF.avg.values, fe_min)
        
    boundReg.fe = boundReg.avg - fe_min
        
    freeBoundRegArea = -(2.4942)*np.log( simps( np.exp(- (boundReg.avg )/(2.4942)),  boundReg[cvname] ) )
    #print(freeBoundRegArea)
    
    return minLocation, minFreeEnergy, minSD, freeBoundRegArea


def old_write_fes_tofile(method, fileName, analyte, probeSMILES, minLocation, minFreeEnergy, minSD):
    
    
    lchain = probeSMILES.count('C')
    
    lfl = probeSMILES.count('F')
    
    lCl = probeSMILES.count('Cl')
    
    lBr = probeSMILES.count('Br')
    
    #print("\nlfl")
    #print(lfl)
    
    X = 'F'
    
    
    if lCl != 0:
        X = 'Cl'
        lfl = lCl
        lchain -= lCl

    if lBr != 0:
        X = 'Br'
        
    lhydrogen = 2*lchain + 2 - lfl
    
    countP = -1
    if probeSMILES.count('P') > 0:
        
        countP = probeSMILES.count('P')
        probeSMILES = probeSMILES.replace('P', 'N')
        
    rdkProbe = Chem.MolFromSmiles(probeSMILES, sanitize=True)
    
    label_i_head_group_h_count = -1
    label_i_head_group_c_count = -1
    if fr_NH2(rdkProbe) == 1 and fr_quatN(rdkProbe) == 0 :
        label_i_head_group_h_count = 2
        label_i_head_group_c_count = 0
        #print(probes[i], "primary amine")
    elif fr_NH1(rdkProbe) == 1 and fr_quatN(rdkProbe) == 0:
        label_i_head_group_h_count = 1
        label_i_head_group_c_count = 1
        #print(probes[i], "secondary amine")
    elif fr_NH0(rdkProbe) == 1 and fr_quatN(rdkProbe) == 0:
        label_i_head_group_h_count = 0
        label_i_head_group_c_count = 2
        #label_i_head_group = "0"
        #print(probes[i], "tertiary amine")
    elif fr_NH0(rdkProbe) == 1 and fr_quatN(rdkProbe) == 1:
        label_i_head_group_h_count = 0
        label_i_head_group_c_count = 3
        #label_i_head_group = ""
        #print(probes[i], "quaternary amine")
    elif fr_NH2(rdkProbe) == 1 and fr_quatN(rdkProbe) == 1 :
        label_i_head_group_h_count = 2
        label_i_head_group_c_count = 1
        #print(probes[i], "secondary amine cation") 
    elif fr_NH1(rdkProbe) == 1 and fr_quatN(rdkProbe) == 1:
        label_i_head_group_h_count = 1
        label_i_head_group_c_count = 2
        #print(probes[i], "tertiary amine cation")
    elif fr_NH0(rdkProbe) == 0 and fr_quatN(rdkProbe) == 1:
        label_i_head_group_h_count = 3
        label_i_head_group_c_count = 0
        #print(probes[i], "primary amine cation")
    
    
    if label_i_head_group_c_count > 0:
        lchain = lchain - label_i_head_group_c_count
        
        
    if countP == -1:
    
        if label_i_head_group_h_count == -1 and label_i_head_group_c_count == -1:
            label_i = 'C' + str(lchain) + X + str(lfl)
        elif label_i_head_group_h_count == 0:
            label_i = 'C' + str(lchain) + X + str(lfl) + '/N' + '(CH3)' + str(label_i_head_group_c_count)
        elif label_i_head_group_c_count == 0:
            label_i = 'C' + str(lchain) + X + str(lfl) + '/NH' + str(label_i_head_group_h_count)
        elif label_i_head_group_h_count > 0 and label_i_head_group_c_count > 0:
            label_i = 'C' + str(lchain) + X + str(lfl) + '/NH' + str(label_i_head_group_h_count) + '(CH3)' + str(label_i_head_group_c_count)
    else:
        
        if label_i_head_group_h_count == -1 and label_i_head_group_c_count == -1:
            label_i = 'C' + str(lchain) + X + str(lfl)
        elif label_i_head_group_h_count == 0:
            label_i = 'C' + str(lchain) + X + str(lfl) + '/P' + '(CH3)' + str(label_i_head_group_c_count)
        elif label_i_head_group_c_count == 0:
            label_i = 'C' + str(lchain) + X + str(lfl) + '/PH' + str(label_i_head_group_h_count)
        elif label_i_head_group_h_count > 0 and label_i_head_group_c_count > 0:
            label_i = 'C' + str(lchain) + X + str(lfl) + '/PH' + str(label_i_head_group_h_count) + '(CH3)' + str(label_i_head_group_c_count)

            
    
    with open(fileName, 'a') as deltaGFileObject:
        
        deltaGFileObject.write(method.upper() + '\t' + analyte.upper()  + '\t' + str(lchain) + '\t' + str(lfl) + '\t' + str(lhydrogen) + '\t' + str(label_i_head_group_h_count) + '\t' + str(label_i_head_group_c_count) + '\t' + label_i + '\t' + str(minLocation) + '\t' + str(minFreeEnergy) + '\t' + str(minSD) +  '\n')


def get_unbiased_property(temperature,  simDir, systemDir, colvarFileList, colvarFileContentList, cvname, cvbiasname, propertyFileList, propertyFileContentList, binding, selectProperty, nbins):
    
    kb = 8.31446261815324*0.001 #kJ/molK
    kbt = kb*temperature  #kJ/mol
    
    preFactor = 2*kbt #kJ/mol for entropy correction
    
    nsamples = len(colvarFileList)
    
    totalDataFromAllSamples = []
    for run in range(nsamples):

        colvarFile = simDir + systemDir + colvarFileList[run]

        # read colvarData into pandas DataFrame
        colvarData = pd.read_csv(colvarFile, delim_whitespace=True, names=colvarFileContentList, comment="#")
        colvarData = colvarData.tail(98300) # remove this after making sure all traj have 100000 frames
        colvarData.reset_index(drop=True, inplace=True)
        
        #print(colvarData)
    
    
        # read property file
        propertyFile = simDir + systemDir + propertyFileList[run]
        
        propertyData = pd.read_csv(propertyFile, delim_whitespace=True, names=propertyFileContentList, comment="#")
        propertyData = propertyData[~propertyData['time2'].str.contains("@", na=False)]
        propertyData = propertyData.astype('float').tail(98300) # sane as above
        propertyData.reset_index(drop=True, inplace=True)
    
        #print(propertyData)
    
        if selectProperty == "netEnergy":
            propertyData['netEnergy'] = propertyData[propertyFileContentList[1:-2]].sum(axis=1)
            
        #print(propertyData['netEnergy'])
                    
        # combine colvarData and propertyData
        combinedData = pd.concat([colvarData, propertyData], axis=1)
        
        totalDataFromAllSamples.append(combinedData.values.tolist())
    
    totalDataFromAllSamplesArray = np.array(totalDataFromAllSamples)
    totalDataFromAllSamplesArrayShape = totalDataFromAllSamplesArray.shape
    #print(totalDataFromAllSamplesArrayShape)
    newshape1 = totalDataFromAllSamplesArrayShape[0]*totalDataFromAllSamplesArrayShape[1]
    newshape2 = totalDataFromAllSamplesArrayShape[2]
    totalDataFromAllSamplesArray = totalDataFromAllSamplesArray.reshape(newshape1, newshape2)
    totalData = pd.DataFrame(totalDataFromAllSamplesArray, columns=combinedData.columns.values.tolist())
    
    
    nameList = ['bound', 'unbound', '']
    # assuming cv is com
    if binding == True:
        totalData = totalData [ (totalData[cvname] <= 1) ]
        dname = nameList[0]
    elif binding == False:
        totalData = totalData [ (totalData[cvname] > 2 ) & (totalData[cvname] <= 3) ]
        dname = nameList[1]
    elif binding == None:
        totalData = totalData
        dname = nameList[2]
                
    # calculate weights of each frame in colvarData
    weights = np.exp( (totalData[cvbiasname]) /kbt)
        
    # get unbiased probabilities
    selectedColumn = totalData[selectProperty]
    p,dz = np.histogram(selectedColumn,range=[selectedColumn.min(),selectedColumn.max()],bins=nbins,weights=weights,density=True)
    dz = dz[:-1]
    
    average = np.average(selectedColumn.values, weights=weights)
    sd = np.sqrt(np.average((selectedColumn.values-average)**2, weights=weights))
    
    pDf = pd.DataFrame(np.array([dz, p]).transpose(), columns=[selectProperty, "probability"])
    
    
    
    return pDf, average, sd
    
    
def get_delta_unbiased_property(averageBoundUnboundList, sdBoundUnboundList):
    
    # assuming 1 is unbound and 0 is bound
    deltaAverage = averageBoundUnboundList[0] - averageBoundUnboundList[1]
    
    # error propagation
    sd = np.sqrt(sdBoundUnboundList[0]*sdBoundUnboundList[0] + sdBoundUnboundList[1]*sdBoundUnboundList[1])
    
    
    return deltaAverage, sd
    

def write_property_tofile(method, fileName, analyte, probeSMILES, propertyAvg, propertySD):
    
    lchain = probeSMILES.count('C')
    lfl = probeSMILES.count('F')
    lhydrogen = 2*lchain + 2 - lfl
    
    
    with open(fileName, 'a') as propertyFileObject:
        
        propertyFileObject.write(method.upper() + '\t' + analyte.upper()  + '\t' + str(lchain) + '\t' + str(lfl) + '\t' + str(lhydrogen) + '\t' + 'C' + str(lchain) + '/F' + str(lfl) + '\t' + str(propertyAvg) + '\t' + str(propertySD) +  '\n')


def get_block_unbiased_property(temperature,  simDir, systemDir, colvarFileName, colvarFileContentList, cvname, cvbiasname, propertyFileName, propertyFileContentList, binding, selectProperty, nbins, startTimeInPS, endTimeInPS):
    
    kb = 8.31446261815324*0.001 #kJ/molK
    kbt = kb*temperature  #kJ/mol
    
    preFactor = 2*kbt #kJ/mol for entropy correction
    
    
    colvarFile = simDir + systemDir + colvarFileName

    # read colvarData into pandas DataFrame
    colvarData = pd.read_csv(colvarFile, delim_whitespace=True, names=colvarFileContentList, comment="#")
    colvarData['time'] = colvarData['time'].round(1)    
    # discard initial few frames [frames where free energy is still being mapped]
    colvarData = colvarData [ (colvarData.time > startTimeInPS) & (colvarData.time <= endTimeInPS) ]
    colvarData.reset_index(drop=True, inplace=True)

    #print(colvarData)
    
    # calculate weights of each frame in colvarData
    weights = np.exp( (colvarData[cvbiasname]) /kbt)
    
    
    # read property file
    propertyFile = simDir + systemDir + propertyFileName
        
    propertyData = pd.read_csv(propertyFile, delim_whitespace=True, names=propertyFileContentList, comment="#")
    propertyData = propertyData[~propertyData['time2'].str.contains("@", na=False)]
    propertyData = propertyData.astype('float')#.tail(98300) # same as above
    propertyData = propertyData [ (propertyData.time2 > startTimeInPS) & (propertyData.time2 <= endTimeInPS) ]
    propertyData.reset_index(drop=True, inplace=True)
    #print(propertyData)
    
    if selectProperty == "netEnergy": #-2
        propertyData['netEnergy'] = propertyData[propertyFileContentList[1:]].sum(axis=1)
        #print("netEnergy ")
        #print(propertyData.iloc[0])

    # combine colvarData and propertyData
    combinedData = pd.concat([colvarData, propertyData], axis=1)
    
    #print(combinedData)
    

    #totalDataFromAllSamples.append(combinedData.values.tolist())
    
    nameList = ['bound', 'unbound', '']
    # assuming cv is com
    if binding == True:
        combinedData = combinedData [ (combinedData[cvname] <= 1) ]
        dname = nameList[0]
    elif binding == False:
        combinedData = combinedData [ (combinedData[cvname] > 2 ) & (combinedData[cvname] <= 3) ]
        dname = nameList[1]
    elif binding == None:
        combinedData = combinedData
        dname = nameList[2]
        
    #print(combinedData)
                
    # calculate weights of each frame in colvarData
    weights = np.exp( (combinedData[cvbiasname]) /kbt)
    
    #print("binding")
    #print(binding)
    #print(weights)
        
    # get unbiased probabilities
    selectedColumn = combinedData[selectProperty]
    p,dz = np.histogram(selectedColumn,range=[selectedColumn.min(),selectedColumn.max()],bins=nbins,weights=weights,density=True)
    dz = dz[:-1]
    
    average = np.average(selectedColumn.values, weights=weights)
    sd = np.sqrt(np.average((selectedColumn.values-average)**2, weights=weights))
    
    pDf = pd.DataFrame(np.array([dz, p]).transpose(), columns=[selectProperty, "probability"])
    
    return pDf, average, sd
