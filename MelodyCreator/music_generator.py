# %%[markdown]

# This script contains multiple functions how to convert our genomes to a playable sound with pyo.
# The functions are based on the tutorial/example from *Kie Codes* that can be found https://github.com/kiecodes/generate-music. 
# %%
from pyo import *

# %%[markdown]

# We define a few variables we will use across our network regarding number of bars, number of notes and bits per note in our generated melodies.

# %%
BITS_PER_NOTE = 4
NUM_BARS = 8 
NUM_NOTES = 4
# %%[markdown]

# This function converts our genome containing bits (0s and 1s) to integers.

# %%
def int_from_bits(bits) -> int:
    return int(sum([bit*pow(2, index) for index, bit in enumerate(bits)]))

# %%[markdown]

# This function converts our genome to a melody dictionary containing all necessary information.

# %%
def genome_to_melody(genome, num_bars: int, num_notes: int, num_steps: int,
                     pauses: int, key: str, scale: str, root: int):
    notes = [genome[i * BITS_PER_NOTE:i * BITS_PER_NOTE + BITS_PER_NOTE] for i in range(num_bars * num_notes)]

    note_length = 4 / float(num_notes)

    scl = EventScale(root=key, scale=scale, first=root)

    melody = {
        "notes": [],
        "velocity": [],
        "beat": []
    }


    for note in notes:
        integer = int_from_bits(note)

        if not pauses:
            integer = int(integer % pow(2, BITS_PER_NOTE - 1))

        if integer >= pow(2, BITS_PER_NOTE - 1):
            melody["notes"] += [0]
            melody["velocity"] += [0]
            melody["beat"] += [note_length]
        else:
            if len(melody["notes"]) > 0 and melody["notes"][-1] == integer:
                melody["beat"][-1] += note_length
            else:
                melody["notes"] += [integer]
                melody["velocity"] += [127]
                melody["beat"] += [note_length]

    steps = []
    for step in range(num_steps):
        steps.append([scl[(note+step*2) % len(scl)] for note in melody["notes"]])

    melody["notes"] = steps
    return melody
# %%[markdown]

# Thi function convert the genome to a list of pyo events, which we can play and listen to. 

# %%
def genome_to_events(genome, num_bars: int, num_notes: int, num_steps: int,
                     pauses: bool, key: str, scale: str, root: int, bpm: int):
    melody = genome_to_melody(genome, num_bars, num_notes, num_steps, pauses, key, scale, root)

    return [
        Events(
            midinote=EventSeq(step, occurrences=1),
            midivel=EventSeq(melody["velocity"], occurrences=1),
            beat=EventSeq(melody["beat"], occurrences=1),
            attack=0.001,
            decay=0.05,
            sustain=0.5,
            release=0.005,
            bpm=bpm
        ) for step in melody["notes"]
    ]