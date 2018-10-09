
I_0= 0.00258;              %Initial moles of initiator
M_0= 1E6;                  %Initial moles of monomer
T=323.45;
R_lm =0;
Tf=10;
C=MMA_Simulation(I_0, M_0, T, R_lm, Tf);

Data= importdata('Data@10.txt');    %Training Data

