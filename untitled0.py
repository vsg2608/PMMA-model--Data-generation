import matlab.engine
eng = matlab.engine.start_matlab()

#%%

I_0= 0.00258               #Initial moles of initiator
M_0= 1E6*1.0                   #Initial moles of monomer
T=323.45
R_lm =0.0
Tf=10.0

#C=MMA_Simulation(I_0, M_0, T, R_lm, Tf);
eng.addpath(r'C:\Users\Vishesh\Desktop\Workspace\BTP',nargout=0)

C=eng.MMA_Simulation(I_0, M_0, T, R_lm, Tf)

print(C)