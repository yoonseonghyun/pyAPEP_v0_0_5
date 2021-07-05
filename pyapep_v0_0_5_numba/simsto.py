
# %% Importing
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

# %% Global variable
R_gas = 8.3145      # 8.3145 J/mol/K

# %% Class "tank"

class tank:
    def __init__(self, L, A, n_comp, E_balance = True):
        self._L = L         # length (m)
        self._A = A         # Cross-sectional area (m^2)
        self._n_comp=n_comp # number of gas component
        self._required = {'Design':True,
        'adsorbent_info':False,
        'gas_prop_info': False,
        'mass_trans_info': False,}
        if E_balance:
            self._required['thermal_info'] = False
    def __str__(self):
        str_return = '[[Current information included here]] \n'
        for kk in self._required.keys():
            str_return = str_return + '{0:16s}'.format(kk)
            if type(self._required[kk]) == type('  '):
                str_return = str_return+ ': ' + self._required[kk] + '\n'
            elif self._required[kk]:
                str_return = str_return + ': True\n'
            else:
                str_return = str_return + ': False\n'
        return str_return
    def adsorbent_info(self, iso_fn, epsi = 0.3, rho_s = 1000, P_test_range=[0,10], T_test_range = [273,373]):
        T_test = np.linspace(T_test_range[0], T_test_range[1],self._N)
        p_test = []
        for ii in range(self._n_comp):
            p_tmp = P_test_range[0] + np.random.random(self._N)*(P_test_range[1] - P_test_range[0])
            p_test.append(p_tmp)        
        try:      
            iso_test = iso_fn(p_test, T_test)
            if len(iso_test) != self._n_comp:
                print('Output should be a list/narray including {} narray!'.format(self._n_comp))
            else:
                self._iso = iso_fn
                self._rho_s = rho_s
                self._epsi = epsi
                self._required['adsorbent_info'] = True
        except:
            print('You have problem in iso_fn')
            print('Input should be ( [p1_array,p2_array, ...] and T_array )')
            print('Output should be a list/narray including {} narray!'.format(self._n_comp))
    def adsorbent_info(self, iso_fn, epsi = 0.3, rho_s = 1000,P_test_range=[0,10], T_test_range = [273,373]):
        T_test = np.linspace(T_test_range[0], T_test_range[1],6)
        p_test = np.zeros([self._n_comp, 6])
        for ii in range(self._n_comp):
            p_tmp = P_test_range[0] + np.random.random(6)*(P_test_range[1] - P_test_range[0])
            p_test[ii,:]=p_tmp
        try:
            for ii,TT in zip(np.arange(6),T_test):
                iso_test = iso_fn(p_test[:,ii], TT)
            if len(iso_test) != self._n_comp:
                print('Output should be a list/narray including {} narray!'.format(self._n_comp))
            else:
                self._iso = iso_fn
                self._rho_s = rho_s
                self._epsi = epsi
                self._required['adsorbent_info'] = True
        except:
            print('You have problem in iso_fn')
            print('Input should be ( [p1_array,p2_array, ...] and T_array )')
            print('Output should be a list/narray including {} narray!'.format(self._n_comp))
    def gas_prop_info(self, Mass_molar):
        stack_true = 0
        if len(Mass_molar) == self._n_comp:
            stack_true = stack_true + 1
        else:
            print('The input variable should be a list/narray with shape ({0:d}, ).'.format(self._n_comp))
        if stack_true == 1:
            self._M_m = Mass_molar
            self._required['gas_prop_info'] = True
    def mass_trans_info(self, k_mass_transfer, a_specific_surf):
        stack_true = 0
        
        if len(k_mass_transfer) == self._n_comp:
            if np.isscalar(k_mass_transfer[0]):
                order = 1
                self._order_MTC = 1
            else:
                order = 2
                self._order_MTC = 2
            stack_true = stack_true + 1
        else:
            print('The input variable should be a list/narray with shape ({0:d}, ).'.format(self._n_comp))
        if stack_true == 1:
            self._k_mtc = k_mass_transfer
            self._a_surf = a_specific_surf
            self._required['mass_trans_info'] = True

    def thermal_info(self, dH_adsorption,
                     Cp_solid, Cp_gas):
        stack_true = 0
        n_comp = self._n_comp
        if len(dH_adsorption) != n_comp:
            print('dH_adsorption should be ({0:d},) list/narray.'.format(n_comp))
        else:
            stack_true = stack_true + 1
        if len(Cp_gas) != n_comp:
            print('Cp_gas should be ({0:d},) list/narray.'.format(n_comp))            
        else:
            stack_true = stack_true + 1
        if stack_true == 2:
            self._dH = dH_adsorption
            self._Cp_s = Cp_solid
            self._Cp_g = Cp_gas
            self._required['thermal_info'] = True

    

# %% Test the defined class "tank"
t1 = tank(5,0.031416,2)
print(t1)



# %% Test "adsorbent_info"
import pyiast
import pandas as pd
# Zeolite 13X
par_ch4 = [7.26927417, 0.33068804] # Based on mol/kg vs bar
par_n2 = [0.62864572, 7.26379457, 1.47727665, 0.04093633] # Based on mol/kg vs bar
dH_list = [16372.5284,11675.65] ## J/mol
def Lang(p_in,par):
    qtmp = par[0] * par[1]*p_in/(1+par[1]*p_in)
    return qtmp

def DSLa(p_in,par):
    qtmp1 = par[0]*par[2]*p_in/(1+par[2]*p_in)
    qtmp2 = par[1]*par[3]*p_in/(1+par[3]*p_in)
    qtmp_return = qtmp1 + qtmp2
    return qtmp_return
P_tmp = np.linspace(0, 49)
q_ch4 = Lang(P_tmp,par_ch4)
#print(di_ch4)
di_ch4 = {'p':P_tmp,
          'q':q_ch4}
df_ch4 = pd.DataFrame(di_ch4)
#print(df_ch4)
iso0 = pyiast.ModelIsotherm(df_ch4,
                            loading_key='q',pressure_key = 'p',
                            model= 'Langmuir', 
                            param_guess = {'M': par_ch4[0],'K':par_ch4[1]})
q_n2 = DSLa(P_tmp, par_n2)
di_n2 = {
    'p':P_tmp,
    'q':q_n2}
df_n2 = pd.DataFrame(di_n2)
iso1 = pyiast.ModelIsotherm(
    df_n2,
    pressure_key= 'p', loading_key = 'q',
    model = 'DSLangmuir', 
    param_guess = {
        'M1':par_n2[0],
        'M2':par_n2[1],
        'K1':par_n2[2],
        'K2':par_n2[3]
    }
)
print(iso0.params)
print(iso1.params)
def Arrh(T,T_ref, dH):
    ret = np.exp(np.abs(dH)/8.3145*(1/T - 1/T_ref))
    return ret

def iso_mix(P,T,):
    P = np.zeros(2)
    P_norm0 = Arrh(T,180,dH_list[0])*P[0]
    P_norm1 = Arrh(T,180,dH_list[1])*P[1]
    P[0] = P_norm0
    P[1] = P_norm1
    if np.sum(P) > 0:
        x_frac = P/np.sum(P)
    else:
        x_frac = np.zeros(2)
    if x_frac[0] < 0.005:
        q = np.zeros(2)
        q[0] = 0
        q[1] = iso0.loading(P_norm0)
        return q
    elif x_frac[1] < 0.005:
        q = np.zeros(2)
        q[0] = iso1.loading(P_norm1)
        q[1] = 0
        return q
    elif x_frac[0] < 0 and x_frac[1] < 0:
        q = np.zeros(2)
        return q
t1.adsorbent_info(iso_mix,)
print(t1)

# %% gas property
molar_mass = [0.016, 0.028]     # kg/mol 
t1.gas_prop_info(molar_mass)
print(t1)


# %% mass transfer
k_mass_trans = [
    [0.00001, 0.00000001],  # [1st: m/sec, 2nd: (m kg)/(sec mol)]
    [0.00001, 0.00000001]]  # [1st: m/sec, 2nd: (m kg)/(sec mol)]
Surface_A_per_V = 1.5E9     # (1.5 x 10^9) m^2/m^3
t1.mass_trans_info(k_mass_trans,Surface_A_per_V)
print(t1)

# %% thermal information
dH_list = [16372.5284, 11675.65]# J/mol
T_ref_list = [180,180]          # K
Cp_g = np.array([40.63,29.22])  # Gas heat capacity: J/mol/K
Cp_s = 948                      # Solid heat capacity: J/kg/K
t1.thermal_info(dH_list,Cp_s, Cp_g) 

print(t1)

# %% thermal information



