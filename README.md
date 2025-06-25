# Bayesian Filter Simulator

Hello! This is a project I made that models various recursive filters in a discrete time approach (extended kalman filter, SIS filter, approximated grid-based approach) within a 2d sim. 

## Problem Statement
The goal is to model the position of a dot on the screen who's true location is known with "uncertainty" (assuming a robotics application this 
would be stuff like noise in the sensor data, inaccuracies in the hardware, wheel slippage ...). 
The problem arises of estimating the state of the dot (or more generally, the system) at a certain discrete time step
given the previous state and new observations on the environment. The goal of the recursive filter is at each time step to propagate the state at the previous time step through system dynamics (i.e. equations modeling changes in the system's state) to arrive at an estimated distribution of the system state called the prior, before modifying the prior with the latest observation 
to attain a more promising prediction. This modified prediction is the posterior of the system, given as a conditional distribution of the current system state given 
the set of all observations up to time step k. By the discrete time step nature of the problem, the observations and states are random processes
consisting of sets of random vectors indexed by time. The special property of these recursive systems is that we are recursively generating the random vector denoting the state at each time step k using only the previous state and a new observation vector - 
that is, we only need information of the previous time step and current data to arrive at the new result. Thus, it makes these algorithms a very efficient and elegant solution 
to the state estimation problem. 

## Project Structure and Key Concepts
- State Space: 2 dimensional, describes relative x and y pixel coordinates from origin at top left of screen [x y]
- Control Input: 2 dimensional, describes the velocity input by the user [vx, vy]
- Sensor Space: x dimensions, describes distance of xth sensor to true position
- System Dynamics: The initial state plus the velocity input over change in time. Extra noise is injected to better model real scenarios.
- Filters:
  - Extended Kalman Filter
      - Applies Kalman Filter methods to the nonlinear observation model, which is a square root function
      - Compute bayesian fusion and generate a noise-dependent confidence scaling using linear algebra machinery 
  - Particle Filter
      - Prior as the importance distribution
      - Systematically resample 
      - Monte Carlo method for determining the state
  - Approximated Grid-Based Filter
      - Approximate (somewhat) continuous pixel values as a discretized finite set of states, then apply the optimal bayesian update rules
   
### Extended Kalman Filter
The extended kalman filter is a generalization of the kalman filter algorithm to nonlinear state transition and observation models. However, we maintain the assumptions of gaussianity, so the only real benefit is being able to apply the kalman filter to nonlinear models. At each time step the extended kalman filter maintains a "best guess" mean and a covariance that delineates a "cloud" of uncertainty around that mean. We can think of it like a normal distribution but generalized to the multidimensional state space, where we hope to "capture" the true state within this multidimensional cloud of certainty. The methods we use to propagate this mean and uncertainty around that mean is more complex, making use of linear algebra and vector calculus topics. In essence, we are propagating the previous posterior's gaussian through the mechanics of bayesian fusion and add dictate the amount of "fusion" with the Kalman Control, a formula that uses mappings between spaces to compute a relative confidence in what the sensors are telling us. 



### Particle Filter
The particle filter is a special type of recursive filter that attempts to model the posterior with discrete spikes called particles in a Monte Carlo fashion. Each 
particle has an importance weight, which is just a fancy term for the probability of the system having the particular state of the particle. 
We sample new particle states from the importance distribution that makes a best guess then provide corrections to the corresponding weight with a ratio between an evaluation
of the posterior (or a function similar up to proportionality) over the importance density with the sampled particle, previous particle state, and new sensor data as inputs. 

<img src="https://github.com/user-attachments/assets/017e958b-6f67-4230-9afa-0b6751cc9370" alt="image" width="400"/>

The particles are distributed uniformly across the screen. 

<img src="https://github.com/user-attachments/assets/8662d259-0744-4585-8086-7bfcdd2b841e" alt="image" width="400"/>

In an instant, the likelihood function adjust the weights of particles and the resampling process (I use systematic sampling) 
causes the particles to "respawn" around these high density regions. Notice that the particles form a circle around the beacon, 
expected behavior as the beacon only gives distance to the dot but not the orientation of the distance. 

<img src="https://github.com/user-attachments/assets/e7d1fd58-81fd-44c2-a9bb-ef1e1138de4b" alt="image" width="400"/>

Finally, the beauty of the particle filter is shown in full effect when the user makes some inputs. 

<img src="https://github.com/user-attachments/assets/3126c4f3-ee85-48fe-bfeb-4ceee12edc17" alt="image" width="400"/>

The preceding examples resample at every time step, but we can choose to resample only after the effective sample size metric 
falls below a certain threshold. I visualized higher probability particles as being larger. Notice that this leads to some
degeneracy as many particles have a negligible impact on the pdf. 

<img src="https://github.com/user-attachments/assets/c146da04-b008-4c4a-99ad-b35fec934075" alt="image" width="400"/>

We can choose to increase the number of particles at the cost of computational resources. This is with 10000 particles. 
Within the constraints of this particular sim, ~150 particles seems to have the best computation-to-convergence-rate trade off,
but in other applications the particle count may be increased further to reduce error at the cost of efficiency.  

## Performance Metrics

When the time step k attains a threshold T the program terminates and displays a matplotlib plot. The metric I use is RMSE (Root Mean Squared Error). To calculate this I let the true position of the dot be the actual state and use methods to determine an overall "average" position of the various filters. For EKF, I just used the mean. For the PF, I take the expected x and y across all particles. For the GF, I take the expected x and y across the centers of all grid cells. 

<img src="https://github.com/user-attachments/assets/8b81b1c1-d59c-43f8-91d0-6b4a921c5c4c" alt="image" width="400"/>



This is a personal project made during Summer of 2025 developed by me, a rising sophomore studying computer science and mathematics at Rutgers University.

References
- Arulampalam, M. S., Maskell, S., Gordon, N., & Clapp, T. (2002). A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian tracking. IEEE Transactions on Signal Processing, 50(2), 174–188.
