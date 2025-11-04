# app.py
import streamlit as st
import pandas as pd
import numpy as np
import random
from io import StringIO

def make_individual(num_hours, n_programs):
    return [random.randrange(n_programs) for in range(num_hours)]

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


#streamlit ui
st.set_page_config(page_title="GA TV Scheduling", layout="wide")
st.title("Genetic Algorithm — TV Scheduling")

csv_path = "modified_program_ratings.csv"  # ensure this file is in the same folder as app.py
df = pd.read_csv(csv_path)

programs = df['Type of Program'].tolist()
hours = [int(c.split()[-1]) for c in df.columns if c.startswith("Hour")]
hours = sorted(hours)
ratings = df.set_index('Type of Program')[[f'Hour {h}' for h in hours]]

st.sidebar.header("GA Run settings")
pop_size = st.sidebar.number_input("Population size", value=150, min_value=10, max_value=1000, step=10)
generations = st.sidebar.number_input("Generations", value=300, min_value=1, max_value=5000, step=1)
seed_input = st.sidebar.text_input("Random seed (optional)", value="")

st.header("Trial parameters (three trials)")
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Trial 1")
    co_r1 = st.slider("CO_R (Trial 1)", 0.0, 0.95, 0.80, step=0.01)
    mut_r1 = st.slider("MUT_R (Trial 1)", 0.01, 0.05, 0.02, step=0.005)
with col2:
    st.subheader("Trial 2")
    co_r2 = st.slider("CO_R (Trial 2)", 0.0, 0.95, 0.70, step=0.01, key="co2")
    mut_r2 = st.slider("MUT_R (Trial 2)", 0.01, 0.05, 0.03, step=0.005, key="mu2")
with col3:
    st.subheader("Trial 3")
    co_r3 = st.slider("CO_R (Trial 3)", 0.0, 0.95, 0.60, step=0.01, key="co3")
    mut_r3 = st.slider("MUT_R (Trial 3)", 0.01, 0.05, 0.04, step=0.005, key="mu3")

if st.button("Run trials"):
    seed = None
    if seed_input.strip() != "":
        try:
            seed = int(seed_input.strip())
        except:
            seed = None

    trial_params = [
        {"label": "Trial 1", "CO_R": co_r1, "MUT_R": mut_r1},
        {"label": "Trial 2", "CO_R": co_r2, "MUT_R": mut_r2},
        {"label": "Trial 3", "CO_R": co_r3, "MUT_R": mut_r3},
    ]

    results = []
    for t in trial_params:
        best_ind, best_fit = run_ga(
            pop_size=int(pop_size),
            generations=int(generations),
            co_r=t["CO_R"],
            mut_r=t["MUT_R"],
            programs=programs,
            hours=hours,
            ratings=ratings,
            seed=seed,
        )
        schedule_df = pd.DataFrame({
            "Hour": hours,
            "Program": [programs[i] for i in best_ind],
            "Rating": [ratings.loc[programs[i], f'Hour {hours[idx]}'] for idx, i in enumerate(best_ind)]
        })
        t["schedule_df"] = schedule_df
        t["best_fit"] = best_fit
        results.append(t)

    # Display results
    for t in results:
        st.subheader(f"{t['label']}: CO_R={t['CO_R']}, MUT_R={t['MUT_R']}, Fitness={t['best_fit']:.4f}")
        st.dataframe(
            t["schedule_df"]
            .assign(Time=lambda df: df["Hour"].astype(str) + ":00")
            .set_index("Time")[["Program", "Rating"]]
        )
        csv_buf = StringIO()
        t["schedule_df"].to_csv(csv_buf, index=False)
        st.download_button(
            label=f"Download {t['label']} schedule CSV",
            data=csv_buf.getvalue(),
            file_name=f"{t['label']}_schedule.csv",
            mime="text/csv"
        )
