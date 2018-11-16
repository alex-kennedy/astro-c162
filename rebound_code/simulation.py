import json
import os
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

import rebound

M_SUN = 1.988435e30 # kg
M_JUP = 1.89813e27 # kg


def load_planet_system(filename):
    """
    Load a CSV file produced by the data processing scripts and create a system
    consisting of a planets information data frame and host star.

    Args:
        filename (str): path to CSV file
    
    Returns:
        pandas.DataFrame: planets dataframe with columns shown in the function
        pandas.DataFrame: dataframe consistem of host star name and mass

    """
    df = pd.read_csv(filename)

    if len(df['pl_hostname'].unique()) > 1:
        raise Exception('More than one star in this dataset! Deal with that in this code, silly.')

    planets = df[['pl_letter', 'pl_orbper', 'pl_orbsmax', 'pl_orbeccen', 'pl_orbincl', 
    'pl_angsep', 'pl_orbtper', 'pl_orblper', 'pl_bmassj', 'pl_bmassprov']]
    star = df[['pl_hostname', 'st_mass']].iloc[0,:]

    return planets, star


def impute_inclination_mass(planets):
    """
    Impute the inclination of planets with the average inclination of the
    other bodies in the system. We need to check if this is a good approximation
    but it's probably not terrible because planets orbit approximately in plane.

    Args:
        planets (pandas.DataFrame): a dataframe with inclination, mass, mass provenance
            columns
    
    Returns:
        pandas.DataFrame: dataframe of same form as input
    """
    # Impute inclinations
    planets['pl_orbincl'] = planets['pl_orbincl'].fillna(planets['pl_orbincl'].mean())
    
    # Adjust Msini masses
    indices = planets['pl_bmassprov'] == 'Msini'
    planets.loc[indices, 'pl_bmassj'] /= np.sin(planets[indices]['pl_orbincl'] * np.pi / 180)

    # Offset so inclination is 0
    planets['pl_orbincl'] -= planets['pl_orbincl'].mean()

    return planets


def non_dimensionalise_masses(planets, star):
    """
    DON'T USE - THE UNITS ARE INCORRECT

    We set the mass of the star to 1000 base the planet masses off of this value.

    Returns:
        pandas.DataFrame: planets
        pandas.DataFrame: star
        float: conversion factor of 1 unit mass to kg
    """
    m_star = M_SUN * star['st_mass'] # kg
    conversion_factor = m_star / 1000 # no units
    star['st_mass_nondimensioned'] = 1000

    m_planets = planets['pl_bmassj'] * M_JUP
    planets['pl_mass_nondimensioned'] = m_planets / conversion_factor

    return planets, star, conversion_factor


def convert_units(planets):
    """
    We will be using units where G = 1. Specifically where masses are in units
    of masses in M_SUN, distances in AU, and time in yr/2pi. 

    Args:
        planets (pandas.DataFrame): dataframe of planets to change

    Returns:
        pandas.DataFrame: adjusted planets dataframe
    """

    # Change masses from M_JUP to M_SUN
    planets['pl_mass_nondimensioned'] = planets['pl_bmassj'] * M_JUP / M_SUN

    # Change orbital period from days to yr/2pi
    planets['pl_orbper'] *= (2*np.pi / 365.25)

    return planets
    

def calculate_start_positions(planets):
    """
    Essentially to determine the relative locations of planets to each other
    from their periastron information. 

    Args: 
        planets (pandas.DataFrame): dataframe of planets
    
    Returns:
        pandas.DataFrame: dataframe of planets with relative positions added
            (not sure which form this will take yet)

    """
    # Make time of periastron shorter to reduce errors
    planets['pl_orbtper'] -= np.floor(np.min(planets['pl_orbtper']))

    mean_angular_motion = 2*np.pi / planets['pl_orbper']
    mean_anomaly = mean_angular_motion * -planets['pl_orbtper']

    # Fourier expansion from https://en.wikipedia.org/wiki/True_anomaly
    term2 = (2*planets['pl_orbeccen'] - (1/4)*planets['pl_orbeccen']**3) * np.sin(mean_anomaly)
    term3 = (5/4)*planets['pl_orbeccen']**2 * np.sin(2*mean_anomaly)
    term4 = (13/12) * planets['pl_orbeccen']**3 * np.sin(3 * mean_anomaly)

    planets['true_anomaly'] = mean_anomaly + term2 + term3 + term4 # + O(e**4)

    return planets


def prepare_simulation(planets, star, particles, dt=1e-3, max_d=None, integrator='whfast'):
    """
    Create and prepare the Rebound simualation object

    Args:
        planets (pandas.DataFrame): planets dataframe with initial positions included
        star (pandas.DataFrame): host star mass
        particles (numpy.ndarray): particles dataframe with first column being semi major axis,
            second column being true anomaly at start
        dt (float): integration timestep
        max_d (float): maximum distance before escape
        integrator (str): integration algorithm, see REBOUND docs 

    Returns:
        rebound.Simulation: rebound simulation object ready for integration
    """
    sim = rebound.Simulation()

    # Add star
    sim.add(m=star['st_mass'], hash='star')

    # Add planets
    for i, p in planets.iterrows():
        m = p['pl_mass_nondimensioned']
        a = p['pl_orbsmax'] # semi-major axis
        e = p['pl_orbeccen'] # eccentricity
        i = p['pl_orbincl'] * (np.pi / 180) # inclination
        omega_bar = p['pl_orblper'] * (np.pi / 180) # longitude of periastron
        true_anomaly = p['true_anomaly'] * (np.pi / 180)
        big_omega = 0 * (np.pi / 180) # need to write up why this is fine
        pl_hash = p['pl_letter']

        sim.add(m=m, a=a, e=e, inc=i, Omega=big_omega, omega=omega_bar-big_omega, f=true_anomaly, hash=pl_hash)

    for i in range(len(particles)):
        sim.add(m=0, a=particles[i,0], f=particles[i,1], inc=particles[i,2], hash=i)
    
    sim.move_to_com() 
    sim.integrator = integrator
    sim.dt = dt
    sim.N_active = len(planets) + 1 # massive bodies are planets + star
    sim.ri_whfast.safe_mode = 0 # yea im cool
    sim.ri_whfast.corrector = 11

    if max_d:
        sim.exit_max_distance = max_d

    return sim


def new_test_particles(a_min, a_max, delta_inc, n):
    """
    Generates needed values for test particles. Uniform distribution between
    a_min and a_max and true anomaly between 0 and 2pi. 

    Args:
        a_min (float): minimum radius
        a_max (float): maximum radius
        delta_inc (float): plus/minus value for the uniform distribution from which to draw
            test particle inclinations
        n (int): number of particles to generate
    Return:
        np.ndarray: array of dimensions (n, 2)
    """
    particles = np.array([
        np.random.uniform(a_min, a_max, n),
        np.random.uniform(0, 2*np.pi, n),
        np.random.uniform(-delta_inc, delta_inc, n)
    ])
    return particles.T


def remove_escaped_particles(sim, d_removed_particles=None, max_d=100, progress_bar=None):
    sim.integrator_synchronize()

    coords = np.array([[p.x, p.y, p.z] for p in sim.particles])
    escaped = np.sqrt(np.sum(coords**2, axis=1)) > max_d

    particles_to_remove = []
    [particles_to_remove.append(sim.particles[i].hash) if remove else None for i, remove in enumerate(escaped)]

    for p_hash in particles_to_remove:
        if d_removed_particles is not None:
            p = sim.particles[p_hash]
            d_removed_particles.append([p_hash.value, sim.t, p.x, p.y, p.z, p.e, p.a, p.inc, p.omega, p.Omega, p.f])
            
        sim.remove(hash=p_hash)

    sim.ri_whfast.recalculate_jacobi_this_timestep = 1

    plural = 'particle' if len(particles_to_remove) == 1 else 'particles'
    message = 'Removed {} {} at time = {}'.format(len(particles_to_remove), plural, sim.t)
    progress_bar.write(message) if progress_bar is not None else print(message)


def new_experiment(system_file, a_min, a_max, n_particles, t_final, snapshot_interval, delta_inc, max_d=100, dt=1e-3):
    """
    Behemoth function to run one experiment and save all the results.

    Args:
        system_file (str): planetary system file to open
        a_min (float): minimum radius to put test particles into
        a_max (float): maximum radius to put test particles into
        n_particles (int): number of test particles
        t_final (float): time to integrate up to (days)
        snapshot_interval (float): time interval between archiving simulations
        delta_inc (float): plus/minus value for the uniform starting distribution of test inclinations
        max_d (float): distance at which a particle is removed before archiving a simulation
        dt (float): timestep for integration
    """
    # Set up an integration progress bar
    progress_bar = tqdm(total=t_final)
    def heartbeat(sim):
        progress_bar.update(dt)

    # Make a folder for the experiment
    start_time = datetime.utcnow()
    dir_name = os.path.join('data', 'simulations', start_time.strftime('%Y-%m-%d_%H-%M-%S'))
    os.mkdir(dir_name)

    # Check git version
    version = subprocess.check_output(['git', 'describe', '--always']).strip().decode("utf-8") 

    info = {
        'start_time': dir_name.split('/')[-1],
        'version': version,
        'system_file': system_file,
        'a_min': a_min,
        'a_max': a_max,
        't_final': t_final,
        'complete': False,
        'snapshot_interval': snapshot_interval,
        'dt': dt
    }

    planets, star = load_planet_system(system_file)
    planets = impute_inclination_mass(planets)
    planets = convert_units(planets)
    planets = calculate_start_positions(planets)
    particles = new_test_particles(a_min, a_max, delta_inc=delta_inc, n=n_particles)

    simulation = prepare_simulation(planets, star, particles, dt=dt, max_d=max_d)

    # Save the info dictionary
    with open(os.path.join(dir_name, 'info.json'), 'w') as f:
        json.dump(info, f, indent=4)

    # Save the initial conditions
    try:
        os.mkdir(os.path.join(dir_name, 'init'))
    except OSError:
        pass
    planets.to_csv(os.path.join(dir_name, 'init', 'planets.csv'))
    pd.DataFrame(particles, columns=['a', 'f', 'inc']).to_csv(os.path.join(dir_name, 'init', 'particles.csv'))

    # Start a list for the removed particles
    d_removed_particles = []

    # Finish setting up simulation and get the show on the road
    simulation.heartbeat = heartbeat
    simulation.automateSimulationArchive(os.path.join(dir_name, 'sim_archive.bin'), interval=snapshot_interval)

    # Ensure escaped particles are removed
    while True:
        try:
            simulation.integrate(t_final, exact_finish_time=0)
            break
        except rebound.Escape:
            remove_escaped_particles(simulation, d_removed_particles, max_d, progress_bar)

    progress_bar.close()

    # Save removed particles information
    df_removed_particles = pd.DataFrame(d_removed_particles, columns=['i', 't', 'x', 'y', 'z', 'e', 'a', 'inc', 'omega', 'big_omega', 'f'])
    df_removed_particles.to_csv(os.path.join(dir_name, 'init', 'removed_particles.csv'))

    # Resave info with completion
    info['complete'] = True
    info['time_to_complete'] = str(datetime.utcnow() - start_time).split('.')[0]
    with open(os.path.join(dir_name, 'info.json'), 'w') as f:
        json.dump(info, f, indent=4)


if __name__ == '__main__':
    # Quick example
    # new_experiment('data/manual/hd-219134.csv', 4.9, 4.999, 100, 1000, 10, 5, 1e-3)
    # new_experiment('data/manual/hd-219134.csv', 0.24, 0.37, 10000, 100, 10, 100, 5e-5)

    new_experiment(
        'data/manual/hd-219134.csv',
        a_min=4.9,
        a_max=4.999,
        n_particles=100,
        t_final=1000,
        snapshot_interval=10,
        delta_inc=1 * (np.pi/180),
        max_d=100,
        dt=1e-3
    )