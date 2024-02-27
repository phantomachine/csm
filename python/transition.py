from __future__ import print_function, division, absolute_import
import sys, os
import socket
import time
import numpy as np
import copy
# import matplotlib
# matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from tabulate import tabulate
#from numba import jit, autojit, double, float64
#import numba
from mpi4py import MPI
import scipy.stats as stats

#-- Custom Classes and Modules ----
from cssegmod import basemod as model   # Model class
from set_batch_parameters import *      # Define parameter combinations here
from setoutpaths_batch import *         # Define output path for saving files
import send                             # for sending auto email

# ================= USER DEFINE BATCH EXPERIMENT SETTINGS HERE ================
# Set or create directory
homename = "dynamics"
homedir = homename + '/'
if not os.path.exists(homedir):
    os.makedirs(homedir)
# # Patch to results of current run
# rootdir = homedir + batch + '/'

# Load previous results?
LOAD_OLD = True
FRESH_GUESS = None
SAVE_RESULTS = True
# Store sequence of approximation results? (Set to FALSE to speed up!)
STORE_APPROXSEQ = False

# Do you want to bug your coauthors with email notifications?
SENDMAIL = False

# x-axis variable (TAU or INFLATION)
PLOT_xaxis_TAU = True

# Use DM good as basis for CEV calculation?
DM_GOOD_CEV = False

# Specify email receipients here:
email_list = ["tcy.kam@gmail.com"]

# MPI rank and cores:
rank = MPI.COMM_WORLD.rank
num_cpus = MPI.COMM_WORLD.size

# Custom setting for defining grids in self.m_grid
ncpu_custom = num_cpus

# Number of equilibrium points
# N_eqm = 24

# Number of Monte-Carlo paths (also number cross-section individuals)
# Hsim = 1000

# ================= EXTERNALLY CALIBRATED PARAMETERS HERE ==============
beta_freq = 'Q'

# Inflation and interest rates from annual data
# Lucas and Nicolini, annual 3-mth T-bill-rate mean = 3.4% p/a
# FRED CPI inflation rate 3.9% p/a
# Sample: 1915 to 2007
tau_data_annual = 0.034329539689549854
i_data_annual = 0.03854204301075268
beta_annual  = (1. + tau_data_annual)/(1. + i_data_annual)
if beta_freq == 'Q':
    tau_data = (1.0 + 0.034)**0.25 - 1.0
    i_data = (1.0 + 0.039)**0.25 - 1.0
elif beta_freq == 'A':
    tau_data = tau_data_annual
    i_data = i_data_annual
beta = (1. + tau_data)/(1. + i_data)

# override experiment settings: User customisable

alpha = 0.0
# V Calibration: JS, 2018-08-07 ------------------------------------------------
# UbarDM = 404.25 # 406.0
# chi = 0.032
# kappa = 0.00399765 # 0.00399765
CALIBRATE = 'tokyo'

if CALIBRATE=='tokyo':
    # Calibration: JS, 2018-08-07 -----------------------------------------
    UbarDM = 407.7725
    chi = 0.03
    kappa = 0.0039970
elif CALIBRATE=='canberra':
# Calibration -- 06/10/2018 -----------------------------------------------
    UbarDM = 406.211700000
    chi = 0.021000000
    kappa = 0.004997650

# Max. time horizon for reaching new steady state
T_max = 300

# Tatonnement smoothing parameter (updating ω sequence)
ϕ = 0.4

# Max iteration of tatonnement process
Iter_transition_max = 50

# Approximation scheme -- see def Interp1d in src/mssmod9b.py
Interpolant = 'slinear'

# Instatiate Model Class
klmod = model(  BETA = beta, UbarCM=1.0, UbarDM=UbarDM, KAPPA=kappa,
            CHIprop = chi, ALPHA = alpha, A = 1.0,
            SIGMA_CM = 2.0, SIGMA_DM = 1.01, TAU = 0.0,
            m_gridsize_ltk = 24, # card.grid m < KAPPA, NCPU-divisible
            m_gridsize_gtk = 48, # card.grid m > KAPPA, NCPU-divisible
            m_gridsize_fine=300, # choice set, table-lookup: initial, crude
            m_gridsize_refine=1e4, # choice set, table-lookup: later, finer
            MAXITER1=3000,     # iterations over price (OMEGA)
            MAXITER2=150,      # iterations over Bellman functionals
            SMOOTH=0.9,        # secant algorithm smoother: omega loop
            TOL0=1e-4,         # %-precision, %-change omega convergence
            TOL1=1e-5,         # Precision for convergence value functions
            TOL2=1e-5,         # Precision for identifying IsExist(lottery)
            Tsim=10000,        # length of monte carlo sims
            DirtyHack=True,    # check exist lotteries up TOL2 Precision
            InterpMethod=Interpolant, # basis function class, interpolant
            TESTMODE=False,     # Developer's troubleshoot setting
            ShortSell = True, # Assumption on borrowing, fixed cost CHI
            ymax_adhoc = 0.00, # Relevant only to ad-hoc borrowing limit
            NCPU = ncpu_custom, RANK = rank
        )


# Begin stopwatch 
tic = time.time()

# ========================================================================
#           STEP 0. Initial SME at τ = 0.0
# ========================================================================

# File path and name
filename = homedir + 'results_init'

# Solve instance of parameterized model - Initial SME
results_init = klmod.SolveSteadyState(   
                                filename, 
                                LOAD_OLD=LOAD_OLD,
                                HOMOTOPY=None,
                                FRESH_GUESS=FRESH_GUESS,
                                SAVE_RESULTS=SAVE_RESULTS,
                                STORE_APPROXSEQ = STORE_APPROXSEQ,
                                )

if (rank == 0):
    if (SAVE_RESULTS==True) and (LOAD_OLD==False or (LOAD_OLD==True and FRESH_GUESS==True)):
        print("\nNow saving results to .NPY file ...")
        np.save(filename + '.npy', results_init)

# ========================================================================
#           STEP 1. Terminal SME at τ = 0.025 (10% p.a.)
# ========================================================================

# File path and name
filename = homedir + 'results_term'

# Solve instance of parameterized model - New/Terminal SME
klmod_new = copy.deepcopy(klmod)
klmod_new.TAU = 0.025
results_term = klmod_new.SolveSteadyState(   
                                filename, 
                                LOAD_OLD=LOAD_OLD,
                                HOMOTOPY=None,
                                FRESH_GUESS=FRESH_GUESS,
                                SAVE_RESULTS=SAVE_RESULTS,
                                STORE_APPROXSEQ = STORE_APPROXSEQ,
                                )

if (rank == 0):
    if (SAVE_RESULTS==True) and (LOAD_OLD==False or (LOAD_OLD==True and FRESH_GUESS==True)):
        print("\nNow saving results to .NPY file ...")
        np.save(filename + '.npy', results_term)


# ========================================================================
#           STEP 2. Initial guess - sequence of prices ω_{t} 
# ========================================================================
if (rank == 0):
    # Get initial ω
    ω_init = results_init['distro']['omega']
    # Get terminal ω
    ω_term = results_term['distro']['omega']
    print("\n\tInitial real wage: %6.5f" %(ω_init))
    print("\n\tTerminal real wage: %6.5f" %(ω_term))

    # Construct initial guess of path { ω(t): t = 0, ..., T_max }
    slope = (ω_term - ω_init)/T_max
    ω_fun = lambda t: ω_init + slope*t # affine path ω_init --> ω_term 
    T_set = np.arange(0, T_max+1, 1)   # Time domain
    ω_path = ω_fun(T_set)              # initial price-path guess
    ω_path_new = ω_path.copy()         # storage for updated price-path
    history_path = []                  # storage for distribution path

# ========================================================================
#           STEP 3. Backward induction from T_max, assuming ω_path
# ========================================================================
if (rank == 0) and (num_cpus==1):
    # Currently only available in serial computation

    # Store history of search paths
    ω_path_search = []
    distro_path_search = []
    value_path_search = []
    policy_path_search = []

    # Loop of guesses of transition path
    for idx_transition in range(Iter_transition_max):
        print("\n----------------------------------------------------")
        print("\nIteration %i of %i ... Transition path guess" 
                        %(idx_transition, Iter_transition_max)
             )
        # Backward induction step
        # Policy change: τ → τ_new ⟹ klmod() → klmod_new()
        print("\n\tBackward induction ... please wait!")
        value_transit = []
        policy_transit = []
        Vbarfit_next = results_term['valuefit']['Vbar']
        omega_term = results_term['distro']['omega']
        
        # Backward induction loop ...
        for t in range(T_max, -1, -1):
            # Stage-t prices
            omega_t = ω_path[t]
            # Under τ_new → klmod_new() ...
            # Step 3.1: Solve Worker problem  
            W,Wfit,lstar,ystar = klmod_new.WorkerProblem(Vbarfit_next,
                                        omega_t, SolMethod='continuous')
            # Step 4.2: Solve Buyer problem
            B,Bfit,bstar,qstar,xstar = klmod_new.BuyerProblem(Vbarfit_next,
                                            omega_t, SolMethod='discrete')
            # Step 3.3: Ex-ante problem in mixed strategies
            if (rank == 0):
                # Solve ex-post participation problem
                V_tilde = klmod_new.Vtilde(Bfit, Wfit, omega_t)
                # Solve induced ex-ante lottery over participation options
                V,Vfit,lottery_supports,lottery_payoffs=klmod_new.V(V_tilde)
                # Define Vbar
                Vbar, Vbarfit = klmod_new.Vbar(V,W)
                # Store value functions
                value_dict = {  'lottery_supports' : lottery_supports,
                                'Vfit' : Vfit,
                                'Wfit' : Wfit,
                                'Bfit' : Bfit,
                                'Vbarfit' : Vbarfit,
                                }
                value_transit.append(value_dict)

                # Step 3.4: Store policy functions
                policy_dict = { 'ystar' : ystar,
                                'lstar' : lstar,
                                'bstar' : bstar,
                                'xstar' : xstar,
                                'qstar' : qstar,
                                }
                # Stage t = T-1, ..., 0 policies (store in reverse-date list)
                policy_transit.append(policy_dict)

            # Step 3.5: one more stage back, repeat until t=0
            Vbarfit_next = Vbarfit

        # ======================================================================
        #           STEP 4. Forward simulation + Generate update on ω_path 
        # ======================================================================
        print("\tNow updating price path ... please wait!")
        for stages_to_go in range(T_max, -1, -1):
            # Current stage t = 0, 1, ..., T_max
            t = T_max - stages_to_go
            # Approximate cross section using LLN for given guess ω_t
            history,flag = klmod_new.Distribution(
                                        value_transit[t]['lottery_supports'],
                                        value_transit[t]['Vfit'],
                                        value_transit[t]['Wfit'],
                                        value_transit[t]['Bfit'],
                                        policy_transit[t]['ystar'],
                                        policy_transit[t]['bstar'],
                                        policy_transit[t]['qstar'],
                                        policy_transit[t]['xstar'],
                                        ω_path[t],
                                        T=1e5,seed=45823
                                        )
            # Update ω_t with new market clearing guess:
            ω_path_new[t] = klmod_new.MarketClearing(history)
            history_path.append(history)

        # Store history of search/guesses of transition path
        ω_path_search.append(ω_path)              # Price path
        distro_path_search.append(history_path)   # Simulated distribution path
        value_path_search.append(value_transit)   # value functions path
        policy_path_search.append(policy_transit) # policy functions path
        # ======================================================================
        #       STEP 5. Check if date-T distribution matches terminal SME dist  
        # ======================================================================
        # Stopping rule 
        distance = np.absolute((ω_path_new[-1]-ω_term)/ω_term)
        print("\tω_term guess error = %6.4g" %(distance))

        if (distance < 0.0036):     # Error tol. less than 0.5%-difference
        # if (distance < 0.005):     # Error tol. less than 0.1%-difference
            break
        if (idx_transition==Iter_transition_max-1):
            print("\n\tMaximum iteration reached!")

        # Tatonnement algorithm: Update for next ω_path
        ω_path = ω_path + ϕ*(ω_path_new - ω_path)

    # End stopwatch
    toc = time.time()
    print("\nTRANSITION.PY: Total run time = %5.6f seconds" %(toc - tic))

    # =========================================================================
    #          Save transition path results to .NPY
    # ===========================================================================
    # The last (-1) element of each 'key' would be the convergent result!
    results_transition = {  'price'  : ω_path_search,
                            'value'  : value_path_search,
                            'policy' : policy_path_search,
                            'distro' : distro_path_search,
                         }
    if (SAVE_RESULTS==True) and (LOAD_OLD==False or (LOAD_OLD==True and FRESH_GUESS==None)):
        print("\nNow saving results to .NPY file ...")
        np.save(homedir + 'results_transition' + '.npy', results_transition)

# =========================================================================
#          Wrapping up
# =========================================================================
if rank==0:
    print("\n\n\nView figures in: " + homedir)
    print("\nFinishing up ... GOODBYE!\t\t\t")
    
if rank==0 and num_cpus > 1:
    # Send an automatic email to subscribers: see SEND.PY
    if SENDMAIL:
        send.message(homedir, custom_message="COMPLETED > ", \
                                                    toaddr=email_list )
    # Kill all processes and exit Python
    MPI.COMM_WORLD.Abort()