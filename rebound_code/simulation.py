import json
import os
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

import rebound

try:
    from IPython.display import display
except ImportError:
    pass

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


def prepare_simulation(planets, star, particles, dt=1e-3, integrator='whfast'):
    """
    Create and prepare the Rebound simualation object

    Args:
        planets (pandas.DataFrame): planets dataframe with initial positions included
        star (pandas.DataFrame): host star mass
        particles (numpy.ndarray): particles dataframe with first column being semi major axis,
            second column being true anomaly at start
        dt (float): integration timestep
        integrator (str): integration algorithm, see REBOUND docs 

    Returns:
        rebound.Simulation: rebound simulation object ready for integration
    """
    sim = rebound.Simulation()

    # Add star
    sim.add(m=star['st_mass_nondimensioned'])

    # Add planets
    for i, p in planets.iterrows():
        m = p['pl_mass_nondimensioned']
        a = p['pl_orbsmax'] # semi-major axis
        e = p['pl_orbeccen'] # eccentricity
        i = p['pl_orbincl'] * (np.pi / 180) # inclination
        omega_bar = p['pl_orblper'] * (np.pi / 180) # longitude of periastron
        true_anomaly = p['true_anomaly'] * (np.pi / 180)
        big_omega = 0 * (np.pi / 180) # wtf, no idea. maybe from ang sep??? Maybe we can just assume it to be zero??

        sim.add(m=m, a=a, e=e, inc=i, Omega=big_omega, omega=omega_bar-big_omega, f=true_anomaly)

    for i in range(len(particles)):
        sim.add(m=0, a=particles[i,0], f=particles[i,1])
    
    sim.move_to_com() 
    sim.integrator = integrator
    sim.dt = dt
    sim.N_active = len(planets) + 1 # massive bodies are planets + star
    sim.ri_whfast.safe_mode = 0
    sim.ri_whfast.corrector = 11

    return sim


def new_test_particles(a_min, a_max, n):
    """
    Generates needed values for test particles. Uniform distribution between
    a_min and a_max and true anomaly between 0 and 2pi. 

    Args:
        a_min (float): minimum radius
        a_max (float): maximum radius
        n (int): number of particles to generate
    Return:
        np.ndarray: array of dimentions (n, 2)
    """
    particles = np.array([
        np.random.uniform(a_min, a_max, n),
        np.random.uniform(0, 2*np.pi, n)
    ])
    return particles.T


def new_experiment(system_file, a_min, a_max, n_particles, t_final, snapshot_interval, dt=1e-3):
    """
    Behemoth function to run one experiment and save all the results.

    Args:
        system_file (str): planetary system file to open
        a_min (float): minimum radius to put test particles into
        a_max (float): maximum radius to put test particles into
        n_particles (int): number of test particles
        t_final (float): time to integrate up to (days)
        snapshot_interval (float): time interval between archiving simulations
        dt (float): timestep for integration
    """
    # Set up an integration progress bar
    progress_bar = tqdm(total=t_final*(t_final/2)/dt)
    def heartbeat(sim):
        progress_bar.update(sim.contents.t)

    # Make a folder for the experiment
    start_time = datetime.utcnow()
    dir_name = os.path.join('data', 'simulations', start_time.strftime('%Y-%m-%d %H-%M-%S'))
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
    planets, star, mass_conversion = non_dimensionalise_masses(planets, star)
    planets = calculate_start_positions(planets)
    particles = new_test_particles(a_min, a_max, n_particles)

    simulation = prepare_simulation(planets, star, particles, dt)

    info['mass_conversion'] = mass_conversion

    # Save the info dictionary
    with open(os.path.join(dir_name, 'info.json'), 'w') as f:
        json.dump(info, f, indent=4)

    # Save the initial conditions
    try:
        os.mkdir(os.path.join(dir_name, 'init'))
    except OSError:
        pass
    planets.to_csv(os.path.join(dir_name, 'init', 'planets.csv'))
    pd.DataFrame({'a': particles[:,0], 'f': particles[:,1]}).to_csv(os.path.join(dir_name, 'init', 'particles.csv'))

    # Finish setting up simulation and get the show on the road
    simulation.heartbeat = heartbeat
    simulation.automateSimulationArchive(os.path.join(dir_name, 'sim_archive.bin'), interval=snapshot_interval)
    simulation.integrate(t_final)

    progress_bar.close()

    # Resave info with completion
    info['complete'] = True
    info['time_to_complete'] = str(datetime.utcnow() - start_time).split('.')[0]
    with open(os.path.join(dir_name, 'info.json'), 'w') as f:
        json.dump(info, f, indent=4)


if __name__ == '__main__':
    # Quick example
    new_experiment('data/manual/hd-219134.csv', 0.24, 0.37, 10, 1000, 10, 1e-3)
