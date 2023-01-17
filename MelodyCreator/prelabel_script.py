# %%[markdown]

# # Pre-Label

# To be able to quickly label a few random generated melodies without running the whole network, we can use this script.
# We will store th labeled melodies in our CSV and therefore can use it once the network is started to pre-train our evaluation network.
# %%
import random
import csv
from music_generator import *
# %%[markdown]

# Let's again simply define a few variable to set how we want to listen to the melodies e.g. key, scale, bpm.
# Additional we need the genome size and the file we want to store our labeled data in.

# %%
num_steps = 1
pauses = True
key = 'C'
scale= 'major'
root = 4 
bpm = 128
GENOME_SIZE = NUM_BARS * NUM_NOTES * BITS_PER_NOTE
LABELED_FILE = "data.csv"

# %%[markdown]

# We need to start the pyo server.

# %%
s = Server().boot()
# %%[markdown]

# Now we're able to generate a random genome, convert it to  melody we can play, rate it and store this label.

# %%
for _ in range(10):
    genome = random.choices([0,1],k = GENOME_SIZE)
    events = genome_to_events(genome, NUM_BARS, NUM_NOTES, num_steps, pauses, key, scale, root, bpm)
    for e in events:
        e.play()
    s.start()


    rating = input("Rating (0-5)")

    for e in events:
        e.stop()
    s.stop()

    with open(LABELED_FILE, 'a') as myFile:
        writer = csv.writer(myFile)
        writer.writerow((tuple(genome),int(rating)))



