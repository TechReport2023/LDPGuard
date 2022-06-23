# LDPGuard

This is our implementation for the paper:

LDPGuard: Defenses against Data Poisoning Attacks to Local Differential Privacy Protocols

LDPGuard is implemented with Python 3.10.

# Environments

Python 3.10

PyCharm

# Dataset

1. IPUMS.  Ruggles Steven, Flood Sarah, Goeken Ronald, Grover Josiah, Meyer Erin, Pacas Jose, and Sobek Matthew. IPUMS USA: Version 9.0 [dataset]. minneapolis, mn: Ipums, 2019. https://doi.org/10.18128/D010.V9.0, 2019.
 
2. Fire.   San francisco fire department calls for service. http://bit.ly/336sddL, 2019.

3. Zipf. Zipf is a synthetic dataset that follows the Zipf distribution.  

# Implementations and Running examples of Baselines

1. We implement state-of-the-art LDP Protocols, kRR, OUE and OLH (detailed in kRR.py, OUE.py and OLH.py).

For each protocol, we implement the encode, perturb and aggregate functions.  

2. In addition, we implement detection methods for OUE and OLH in OUE.py and OLH.py.

3. By running main.py with Pycharm, we can obtain the results of baselines.


# Implementations  and Running examples of LDPGuard

1. We implement LDPGuard in LDPGuard.py. We can run it with Pycharm to evaluate the performance of LDPGuard.

2. We can evaluate LDPGuard on different dataset by setting  distribution = 'XXX' in the main function. For example, distribution = 'IPUMS'. 

3. Set Beta = 0.01 or 0.05 or 0.1 to evaluate the effect of Beta (i.e., percentage of fake users).  

4. Set NumofTarget = 5 or 10 or 15 to evaluate the effect of r (i.e., the number of targets). 

5. Set epsilon = 1 or 1.5 or 2 to evaluate the effect of epsilon (i.e, privacy budget)

