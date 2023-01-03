## Analog CMOS Optimization Library
# Author: Robert Block
#
# License Information
# FREEWARE LICENSE AGREEMENT
# Derived from the Analog CMOS Design, Tradeoffs, and Optimization
# design spreadsheet. Copyright 2000-2008 David M. Binkley

import numpy as np

"""
Class optlib
Analog Optimization Library

Usage:
Initialize library with CMOs Process Parameters
Use calculation functions to derive desired results

TODO:
 - [ ] Create a generic search function to replace current iteration loops
 - [ ] Add better comments throughout
 - [ ] Create load function for process parameters
"""
class optlib:

    """
    Optlib initialization
    """
    def __init__(self):
        # Call import scripts here for model parameters
        self.k = 1.3806e-23   # Boltzmann's constant, J/K
        self.q = 1.603e-19    # Electron charge, C
        None

    """
    Function CalcMOSParams
    Calculates process params based on inputs
    """
    def CalcMOSParams(self,Id,ICPrime,Ldrawn,WdrawnEachTrimmed=0,M=1,Vsb=0,fflicker=1,T_C=27):
        self.LoadProcessParams(None)

        if np.round(M) != M:
            print("Warning, M is not an integer. Rounding to the nearest integer.")
            M = np.round(M)
            
        # Check for valid user inputs, raise errors and quit if inputs are invalid
        if Id < 0.000001:
            self.errorHandler("Drain current must be >= 0.000001 uA")
        elif ICPrime < 0.000001:
            self.errorHandler("ICPrime must be >= 0.000001")
        elif Ldrawn < self.Ldrawmin:
            self.errorHandler("Drawn length must be greater than process minimum of {} um",self.Ldrawmin)
        elif M < 1:
            self.errorHandler("Multiplicity must be greater than or equal to 1")
        elif Vsb < 0:
            self.errorHandler("Source-Body voltage must be greater than or equal to 0")
        elif Vsb > self.Vddmax:
            self.errorHandler("Source Body Voltage exceeds process maximum")
        elif fflicker < 0:
            self.errorHandler("Flicker corner must be positive")
        elif (T_C < -200) or (T_C > 200):
            self.errorHandler("Temperature must be between -200 and 200C")
        
        # Set additional global values
        self.T = self.T_C + 273             # Temperature in K
        self.Ut= self.k *self.T / self.q    # Thermal voltage

        # Temperature adjustment of Fermi potential, phi
        self.Eg = 1.16 - 0.000702 * (self.T**2)/(self.T + 1108) # Bandgap energy at T
        self.Eg_Tnom = 1.16 - 0.000707 * (self.Tnom**2)/(self.Tnom + 1108) # bandgap energy at Tnom
        self.phi = self.phi_Tnom * (self.T/self.Tnom) - 2*self.Ut*np.log(self.T/self.Tnom) - self.Eg_Tnom * (self.T/self.Tnom) + self.Eg

        # Temperature adjustment of U0, Ecrit, and Vto
        self.U0 = self.U0_Tnom * (self.T/self.Tnom)**self.Bex
        self.Ecrit = self.Ecrit_Tnom * (self.T/self.Tnom)**self.UCex
        self.Vto = self.Vto_Tnom + self.TCV * (self.T-self.Tnom)
        
        # Composite process parameters
        self.k0 = 0.1 * self.U0 *self.Cox   # uA/V^2
        self.I0 = 2 * self.n0 *self.k0 *self.Ut**2  # Technology current, uA 

        # Effective widths and lengths
        self.L = Ldrawn - self.DL
        self.W = (self.L/ICPrime) * (Id/self.I0)  
        self.WL = self.L * self.W

        # Widths for each finger
        self.Ldrawneach = (self.W/M)+self.DW
        if self.Ldrawneach < self.Ldrawmin:
            print("WARNING: drawn width is below limit.")
        
        # Vt adjusted for Vsb from Vto
        self.Vt = self.Vto + self.gamma *( np.sqrt(self.Vsb+self.phi) - np.sqrt(self.phi) )
        
        # Veff by terative solution
        self.Veff = self.VeffFromId(Id,Vsb,self.W,self.L)

        # Operating value of Vgs
        self.Vgs = self.Vt + self.Veff

        # Test to see if valid
        if (self.Vgs + Vsb) > self.Vddmax:
            self.errorHandler("Bias voltage exceeds process maximum")

        # Find operating value of n
        self.n = self.nFromVeff(self.Veff,Vsb)

        # corrected inversion coefficient using actual operating n vs n0
        self.IC = Id / ((2*self.n*self.k0*self.Ut**2)*(self.W/self.L))

        # Vdsat from IC
        self.Vdsat = 2 * self.Ut * (np.sqrt(self.IC+0.25)+0.5+1)
        ## --------------------- LEFT OFF HERE -----------------------------##
        
    
    """
    Function LoadProcessParams

    Loads process parameters from file.  Todo: automatic load from .scs file

    """
    def LoadProcessParams(self,filename):
        # for now, just load in default values for example 180nm process
        if filename == None:
            self.U0_Tnom = 422
            self.Cox = 8.41
            self.gamma = 0.56
            self.phi_Tnom = 0.85
            self.n0 = 1.35          # average value of substrate factor for IC definition
            self.Vto_Tnom = 0.42         # Zero Vsb threshold voltage
            self.Ecrit_Tnom = 5.6   # Velocity saturation critical electric field (horizontal field)
            self.Vsatexp = 1.3
            self.theta = 0.28       # mobility reduction factor (1/V) for vertical field
            self.DW = 0             # lateral diffusion width
            self.DL = 0.028         # lateral diffusion length
            self.VAL = 6            # Early voltage for channel length modulation (note: should vary with L)
            self.dVtDIBL = -0.008   # Drain induced barrier lowering with Vds at min L
            self.dVtDIBLex = 3      # Exponent in DIBL eqn
            self.kf0 = 3.18e-13     # Flicker noise factor in weak inversion
            self.vkf = 1            # Voltage describing flicker noise increase
            self.af = 0.85          # Flicker noise slope
            self.aVto = 0.005       # V*um, Local area threshold voltage mismatch factor
            self.akp  = 0.02        # um, Local area transconductance mismatch factor
            self.Bex  = -1.5        # Mobility temperature exponent
            self.UCex = 0.8         # Velocity saturation critical electric field temco
            self.TCV = -0.001       # V/K, Temco for Vto
            self.Tnom = 300         # K, nominal temperature
            self.Cgdo_process = 0.8         # fF/um, gate-drain overlap capacitance
            self.Cgso_process = 0.8         # fF/um, gate-source overlap capacitance
            self.Cgbo_process = 0           # fF/um, gate-body overlap capacitance
            self.CJ = 1             # fF/um^2, Drain, source diffision area capacitance
            self.PB = 0.9           # V, Drain, source diffision area capacitance built in potential
            self.MJ = 0.5           # Drain, source diffision area capacitance voltage exponent
            self.CJSW = 0.3         # fF/um, Perimeter capactiance
            self.PBSW = 0.8         # V, Drain, source diffision area perimeter built-in potential
            self.MJSW = 0.33        # Drain, source diffusion area perimeter voltage exponent
            self.Wdifext = 0.5      # Drain, source diffusion exterior width
            self.Wdifint = 0.6      # Drain, source diffusion interior width
            self.Wdrawmin = 0.3     # um, minimum width
            self.Ldrawmin = 0.18    # um, minimum length
            self.Vddmax = 1.8       # V, maximum Vdd




    """
    Function VeffFromId

    Calculates Veff from the drain current through iteration
    """
    def VeffFromId(self,Id, Vsb, W, L):
        Veffstep = 0.0001
        IdReltol = 0.001

        # Find the inversion coefficient in order to obtain starting guess for Veff
        ICFixed = Id / (2 * 1.4 * (0.1 * self.U0 * self.Cox) * (self.Ut**2) * (W/L))
         
        if ICFixed < 0.0003:
            Veff = -0.33
        elif ICFixed < 0.003:
            Veff = -0.25
        elif ICFixed < 0.03:
            Veff = -0.164
        elif ICFixed < 0.3:
            Veff = -0.072
        elif ICFixed < 3:
            Veff = 0.04
        elif ICFixed < 30:
            Veff = 0.23
        elif ICFixed < 300:
            Veff = 0.75
        elif ICFixed < 3000:
            Veff = 2.35
        else:
            Veff = 5

        IdTrial = self.IdFromVeff(Veff,Vsb,W,L)
        IdError = IdTrial - Id
        i = 0

        while (np.abs(IdError/Id)>IdReltol):
            # Compute slope of Id to interpolate for value of Veff
            IdForSlope = self.IdFromVeff(Veff+Veffstep, Vsb, W, L)
            IdSlope= (IdForSlope - IdTrial)/Veffstep

            # Compute new Veff value by moving over error divided by slope
            Veff = Veff - (IdError/IdSlope)

            # Evaluate new Id
            IdTrial = self.IdFromVeff(Veff,Vsb,W,L)
            IdError = IdTrial - Id
            i = i + 1

            if i > 100:
                errText = "VeffFromId failed to converge in 100 iterations"
                self.errorHandler(errText)
                return None
        
        return Veff

    """
    Function gmFromVeff

    Finds gm from Veff by solving for change in Id for small step in Veff
    
    Inputs: 
    Veff = Vgs - Vt (V)
    Vsb = source-body voltage (V)
    W = effective width (um)
    L = effective length (um)

    Outputs:
    gm (uS)
    """
    def gmFromVeff(self,Veff, Vsb, W, L):
        Vgmstep = 0.0001
        IdAbove = self.IdFromVeff(Veff + 0.5*Vgmstep, Vsb, W, L)
        IdBelow = self.IdFromVeff(Veff - 0.5*Vgmstep, Vsb, W, L)
        return (IdAbove - IdBelow)/Vgmstep

    """
    Function FindVindif1dB
    Finds the 1dB compression for a differential pair
    """
    def FindVindif1dB(self,Id, Veff, gm, Vsb, W, L):
        IdDifReltol = 0.001

        # Starting guess for Vindif1dB
        Vindif = Id / gm
        IdDif1dB = Vindif * gm * 0.89125

        Veff2 = self.Veff2DeffPair(Id,Veff,Vindif,Vsb,W,L)
        Id1 = self.IdFromVeff(Veff2 + Vindif, Vsb, W, L)
        Id2 = self.IdFromVeff(Veff2, Vsb, W, L)
        IdDif = Id1 - Id2
        i = 0 # number of iterations

        while ((IdDif < IdDif1dB*(1 - IdDifReltol)) or (IdDif > IdDif1dB*(1 + IdDifReltol))):
            Vindif = Vindif - (IdDif1dB - IdDif)/gm
            Veff2 = self.Veff2DeffPair(Id,Veff,Vindif,Vsb,W,L)
            Id1 = self.IdFromVeff(Veff2 + Vindif, Vsb, W, L)
            Id2 = self.IdFromVeff(Veff2, Vsb, W, L)
            IdDif = Id1 - Id2
            i = i + 1

            if i > 100:
                errText = "FindVindif1dB could not find result in 100 iterations"
                self.errorHandler(errText)
                return None
        
        return Vindif

    """
    Function Veff2DiffPair
    Iteratively finds Veff in the second device of a differential pair
    """
    def Veff2DeffPair(self,Id,Veff,Vindif, Vsb, W, L):
        IdSumIdeal = 2 * Id
        IdSumReltol = 0.001
        Veffstep = 0.0001

        #Initial guess
        Veff2 = Veff - Vindif /2
        Id1 = self.IdFromVeff(Veff2+Vindif,Vsb,W,L)
        Id2 = self.IdFromVeff(Veff2,Vsb, W, L)
        IdSum = Id1 + Id2   # sum of diff pair current in uA
        IdSumError = IdSum - IdSumIdeal
        i = 0

        while(np.abs(IdSumError/IdSumIdeal)>IdSumReltol):
            Id1Slope = self.IdFromVeff(Veff2 + Vindif + Veffstep, Vsb, W, L)
            Id2Slope = self.IdFromVeff(Veff2 + Veffstep, Vsb, W, L)
            IdSumSlope = (Id2Slope + Id1Slope -Id2 - Id1) / Veffstep
            # Compute new Veff2
            Veff2 = Veff2 - (IdSumError / IdSumSlope)
            # Evaluate new values
            Id1 = self.IdFromVeff(Veff2+Vindif,Vsb,W,L)
            Id2 = self.IdFromVeff(Veff2,Vsb, W, L)
            IdSum = Id1 + Id2   # sum of diff pair current in uA
            IdSumError = IdSum - IdSumIdeal
            i = i + 1
            if i > 100:
                errText = "Veff2DiffPair could not find Veff2 in 100 iterations"
                self.errorHandler(errText)
                return None
        
        return Veff2

    """
    Function IdFromVeff

    Inputs: 
    Veff = Vgs - Vt
    Vsb (normal reverse biased source, substrate) 
    W, effective width in um
    L, effective length in um

    Global Process Params: 
    u0 (cm^2/Vs), low field mobility
    Cox (fF/um^2), gate oxide capacitance
    gamma (V^1/2), body effect factor
    phi, twice the Fermi potential in V; positive
    Ut, thermal voltage (V)
    Ecrit (V/um), velocity saturation critical field
    Vsatexp, velocity saturation transition exponent
    theta (1/V), mobility reduction coefficient
    theta1, alternate mobility reduction coefficient added with Veff/LEsat

    Model               Ecrit   Vsatexp     theta   theta1
    No vel. sat, VFMR   1E40    1           0       0
    Simple              Ecrit   1           0       theta
    Enhanced            Ecrit   1.3         theta   0

    Output: Id (uA)
    """
    def IdFromVeff(self,Veff,Vsb,W,L):
        theta1 = 0  # Use the enhanced model by default

        # Find operating value of substrate factor, n_
        # differs from global value n
        nLocal = self.nFromVeff(Veff,Vsb)

        # Veffs = (n*Ut/2)*ln(1 + exp(2*Veff/(n*Ut)))
        Veffs = (nLocal * self.Ut /2)*np.log(1+np.exp(2*Veff/(nLocal*self.Ut)))
        
        # Find the specific current
        Ispecific = 2 *nLocal * (0.1 *self.U0 *self.Cox) * (self.Ut**2) * (W/L)
        
        # Adjust specfic current for velocity sat, mobility reduction
        mobDenom = 1 + self.theta * Veffs
        velSatDenom = (1 + ((theta1 + 1/(L*self.Ecrit))*Veffs)**self.Vsatexp)**(1/self.Vsatexp)

        Ispecificadj = Ispecific * (1/mobDenom) * (1/velSatDenom)
        
        return Ispecificadj * (np.log(1+np.exp(Veff/(2*nLocal*self.Ut))))**2

    """
    Function nFromVeff
    Inputs:
    Veff = Vgs - Vt
    Vsb = source-body voltage (positive for nmos and pmos)
    """
    def nFromVeff(self,Veff,Vsb):
        nLocal = 1.4  #starting value for n local
        nNext = 1 + 0.5 * self.gamma / np.sqrt(self.phi+4*self.Ut+Veff/nLocal+Vsb)
        reltol = 0.0001
        i = 0
        while (np.abs(nNext-nLocal)/nNext > reltol):
            nLocal = nNext
            nNext = 1 + 0.5 * self.gamma / np.sqrt(self.phi+4*self.Ut+Veff/nLocal+Vsb)
            i = i +1
            if i > 100:
                errText = "Error: n not found after 100 iterations"
                print(errText)
                self.errorHandler(errText)
                break
        return nNext
    
    """
    Function Error Handler
    Inputs: errText - Text describing nature of error

    Todo: add functionality for criticality of error (full exit or continue)
    """
    def errorHandler(self,errText):
        print("{}, exiting...",errText)
        quit(1)
