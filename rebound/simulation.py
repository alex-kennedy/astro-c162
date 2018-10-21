import rebound
import pandas as pd
import numpy as np


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
    # TODO: the whole function hehe

    return planets


def prepare_simulation(planets, star):
    """
    Create and prepare the Rebound simualation object

    Args:
        planets (pandas.DataFrame): planets dataframe with initial positions included
        star (pandas.DataFrame): host star mass

    Returns:
        rebound.Simulation: rebound simulation object ready for integration
    """
    sim = rebound.Simulation()

    # Add star
    sim.add(m=star.st_mass)

    # Add planets

    return sim


planets, star = load_planet_system('data/manual/hd-219134.csv')
planets = impute_inclination_mass(planets)