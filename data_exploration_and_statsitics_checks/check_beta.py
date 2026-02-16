import numpy as np
import matplotlib.pyplot as plt

#events = np.load("/work/submit/anton100/msci-project/FCC-BB-GenAI/guineapig_raw_trimmed_3/generated_events.npy", allow_pickle=True)
events = np.load("guineapig_raw_trimmed.npy", allow_pickle=True)


beta = []
count = 0

for ev in events:

    beta_ev = ev[:, 1:4]  
    beta.append(beta_ev)

    if np.any(beta_ev > 1) or np.any(beta_ev < -1):
        print("beta out of range:", beta_ev)
        count += len(beta_ev)

print("Total particles with beta out of range:", count)
print("Total particles:", sum(len(b) for b in beta))
