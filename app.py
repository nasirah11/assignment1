# app.py
import streamlit as st
import pandas as pd
import numpy as np
import random
from io import StringIO

def make_individual(num_hours, n_programs):
    return [random.randrange(n_programs) for _ in range(num_hours)]

def fitness(ind, programs, hours, ratings):
    total = 0.0
    for i, prog_idx in enumerate(ind):
        prog = programs[prog_idx]
        hour = hours[i]
        total += ratings.loc[prog, f'Hour {hour}']
    return total

def tournament_selection(pop, fits, k=3):
    selected = random.sample(range(len(pop)), k)
    best = max(selected, key=lambda i: fits[i])
    return pop[best][:]

def single_point_crossover(a, b):
    if len(a) <= 1:
        return a[:], b[:]
    pt = random.randint(1, len(a)-1)
    ca = a[:pt] + b[pt:]
    cb = b[:pt] + a[pt:]
    return ca, cb

def mutate(ind, n_programs):
    i = random.randrange(len(ind))
    ind[i] = random.randrange(n_programs)

def run_ga(pop_size, generations, co_r, mut_r, programs, hours, ratings, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    num_hours = len(hours)
    n_programs = len(programs)
    pop = [make_individual(num_hours, n_programs) for _ in range(pop_size)]
    best_history = []
  for gen in range(generations):
        fits = [fitness(ind, programs, hours, ratings) for ind in pop]
        new_pop = []
        while len(new_pop) < pop_size:
            parent1 = tournament_selection(pop, fits)
            parent2 = tournament_selection(pop, fits)
            if random.random() < co_r:
                child1, child2 = single_point_crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]
            if random.random() < mut_r:
                mutate(child1, n_programs)
            if random.random() < mut_r:
                mutate(child2, n_programs)
            new_pop.extend([child1, child2])
        pop = new_pop[:pop_size]
    final_fits = [fitness(ind, programs, hours, ratings) for ind in pop]
    best_idx = int(np.argmax(final_fits))
    best_ind = pop[best_idx]
    best_fit = final_fits[best_idx]
    return best_ind, best_fit
