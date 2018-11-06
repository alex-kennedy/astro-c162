# Astronomy C162 Project

Repository for Alex Kennedy's and Chingham Fong's final project for C162. 

The goal is to determine if the Kepler 90 system can support an asteroid belt. 

Uses Python 3. It requires the Rebound package, which must be run on a unix system (use Linux subsystem for Windows). 

## Where we're up to

We have the infrastructure for running reproducible simulations and saving the results. A run down of the process so far follows. 

- We process and save all the planetary systems from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu) confirmed planets table
- We load in a planetary system after some manual adjustment (at the moment it's HD 219134 but this can change)
- We impute the inclination by setting it to the average of any known inclinations in the sytem (*needs justification*). The inclinations shifted such that their mean is 0. 
- We non-dimensionalize the mass to reduce floating point error. The mass of the star is set to 1000 and the planets are based off this. 
- We calculate the planets' start positions from their time of periastron. Uses a [Fourier expansion](https://en.wikipedia.org/wiki/True_anomaly) to get the true anomaly at a certain point in time (arbitrary). 
- We generate a set of particles which are uniformly distributed between a minium and maximum semi-major axis and random true anomaly. Inclination and eccentricity are set to 0. 
- Set up the REBOUND simulation object with the star, planets and test particles (asteroids). 
- At this stage, longitude of the ascending node is set to 0, because I can't find out how to determine it! 
- We use the `whfast` algorithm at present.
- Every time we do a simulation (at the moment, I've called these 'experiments') generate an `info.json` file and save the particles and planets in an `init` folder. 
- A simulation archive file called `sim_archive.bin` is generated (see REBOUND docs) with an adjustable time interval.

## To do

- Justify assumptions for inclination of planets
- Justify assumptions for longitude of the ascending node of planets
- Justify use of simulation over analysis with resonances etc.
- Check errors with respect to starting points of planets
- Decide for sure on a planetary system
- Check that the actual planetary system is stable
- Justify choice of time step (there's a paper somewhere)
- Decide on some simulation parameters
    - Final time?
    - How many asteroids?
    - Do we make the asteroids affect the planets?
    - Range of semi-major axes for asteroids?
    - Introduce inclination to asteroids also? 
- All post-processing of simulations
    - What do we mean by stable
    - Apply this definition to our system
    - Do some statistics (do we have 'convergence'?)
    - Make pretty plots
- Get A+