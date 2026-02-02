import numpy as np
import matplotlib.pyplot as plt

tracks = np.load("mc_gen1.npy", allow_pickle=True)

print("Total number of events:", len(tracks))  # Total number of events

total_particles = sum(event.shape[0] for event in tracks)
print("Total number of particles across all events:", total_particles)  # Total number of particles

number_particles = []

for event in tracks:
    number_particles.append(event.shape[0])


plt.hist(number_particles, bins=50, alpha=0.75, color='blue')
plt.savefig("number_of_particles_per_event_histogram.png")


