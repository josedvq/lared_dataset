import os
import pandas as pd
import numpy as np

def load_vad(vad_path):
    vad = {}
    for i in range(1, 45):
        fpath = os.path.join(vad_path, f'{i}.vad')
        if os.path.exists(fpath) and os.path.isfile(fpath):
            vad[i] = pd.read_csv(fpath, header=None).to_numpy()

    if len(vad) == 0:
        print('load_vad called but nothing loaded.')

    return vad


def get_turns(ss):
    turns = []
    indices = np.where(np.diff(ss) != 0)[0]
    prev_index = 0
    for idx in indices:
        if ss[idx] == 1:
            turns.append([prev_index, idx])
        prev_index = idx
    return turns


def get_turn_lengths_per_subject(vad):
    turn_lengths = {}
    for pid, pid_vad in vad.items():
        if pid not in turn_lengths:
            turn_lengths[pid] = []

        turns = get_turns(pid_vad.squeeze())
        for t in turns:
            turn_lengths[pid].append((t[1] - t[0]) / 100)
    return turn_lengths

def get_turn_lengths(vad):
    tl_per_subject = get_turn_lengths_per_subject(vad)
    return [t for tls in tl_per_subject.values() for t in tls]
