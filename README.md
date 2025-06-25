# Bayesian Filter Simulator

Hello! This is a project I made that models various recursive filters in a discrete time approach (extended kalman filter, SIS filter, approximated grid-based approach) within a 2d sim. This repository is meant as a brief overview of these methods for learners and people curious about state estimation in general. 

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
      - Approximate (more or less) continuous pixel values as a discretized finite set of states, then apply the optimal bayesian update rules
   
### Extended Kalman Filter
The extended kalman filter is a generalization of the kalman filter algorithm to nonlinear state transition and observation models. However, we maintain the assumptions of gaussianity, so the only real benefit is being able to generalize KF to more models. At each time step the extended kalman filter maintains a "best guess" mean and a covariance that delineates a "cloud" of uncertainty around that mean. We can think of it like a normal distribution but generalized to the multidimensional state space, where we hope to "capture" the true state within this multidimensional cloud of certainty. The methods we use to propagate this mean and uncertainty around that mean is more complex, making use of linear algebra and vector calculus topics. In essence, we are propagating the previous posterior's gaussian through the mechanics of bayesian fusion and add dictate the amount of "fusion" with the Kalman Control, a formula that uses mappings between spaces to compute a relative confidence in what the sensors are telling us. Actually, everything I've stated prior to this is basically just Kalman Filter, the main difference in the "extended" variation is that we derive a first order taylor approximation of the nonlinear models by calculating the jacobian (generalized derivative) at the prior. This allows us to run the kalman filter updates without changing any formulas as we now have linear approximations of our models. 

<img src="https://github.com/user-attachments/assets/0854ac74-3db7-452f-a84d-316f9633de64" alt="image" width="400"/>

The blue dot is the mean of the gaussian posterior predicted by the EKF. The transparent ellipse is a 2 SD confidence interval dictated by the covariance of the posterior. 

<img src="https://github.com/user-attachments/assets/bb524eff-ef52-49ab-a944-9b255719b790" alt="image" width="400"/>

As expected, increasing the number of sensors increases the precision of the posterior and the accuracy in general. This is the core of bayesian fusion; When we have multiple distributions "agreeing" on a certain point, the uncertainty of the new distribution dramatically decreases. 


### Grid Filter
The grid based filter solves the problem of tractability, meaning feasability to compute, of the theoretical recursive filter. For example, the Chapman-Kolmogorov equation is the idealized way to calculate the prior distribution given the observation and state at the previous time step. However, because this equation takes an integral, we say that is intractable and thus we must compute something "like" that equation, or at least follow the general principles of the optimal filter when pushing along new posteriors. The grid filter basically says rather than taking an integral over all the possible states, which is practically infeasible with continuous values as is usually the case in real applications, let's discretize the state space into a finite set of states and perform the Chapman-Kolmogorov but as a sum over these states. Specified to this simulation, that means taking the state space (which is just x and y pixel coordinates) and chunking it up into discrete, finite cells. Then, we can perform the exact recursive filter updates on the "center" of each state. Of course, this comes with a few problems: The division of the state space must be sufficiently "dense" to model its continuous nature, but in nontrivial cases this comes at tremendous computational cost. This is because in complex problems (high dimensional state spaces, dense representations of the state space) the computation required grows exponentially. For example, the generalized Chapman-Kolmogorov equation to sums takes an overall "flow" into each state from every other state. This means iterating through every state, then deriving a probability of entering that state from every other state. In this simulation this means iterating through all n^2 grid cells and within each iteration iterating through n^2 grid cells, a terrible time complexity of n^4. In fact, when making this simulation it would not even function with cells of size 50px (for reference, the width of the simulation is 1200px) and external optimizations were necessary for the grid filter to even be useable (numba optimizations). Hence, this mode of filtering should only be used for very simple applications or teaching purposes. For practical usage other methods should be used. 

<img src="https://github.com/user-attachments/assets/0309651a-20cb-4812-8d8c-9797610e2785" alt="image" width="400"/>

Upon start of simulation, the weights converge to the cells that maximize likelihood, i.e. where the beacon says the system could be at. Because its a distance sensor and does not track orientation, a circle is expected. Note that I initialize the cells as a gaussian about the true start position, which explains the uneveness in the circle. 

<img src="https://github.com/user-attachments/assets/3ebfe70c-2551-492a-a131-eb067151a640" alt="image" width="400"/>

As expected, upon movement the prediction refines and the distribution converges around the true position. 

<img src="https://github.com/user-attachments/assets/f9272784-f16f-4b31-aae4-7ae03becb040" alt="image" width="400"/>

We can also choose to refine the size of how we divide the state space, i.e. increasing the grid's resolution, at the cost of computational effort. 



### Particle Filter
The particle filter is a special type of recursive filter that attempts to approximate the posterior with discrete spike "particles" using monte carlo methods. The particles have attached weights, giving the evaluation of the approximated distribution at the state of the particle. In essence, we're throwing guesses at a distribution and weighing the good guesses, which when we add more particles should better model that distribution. At each recursive step we sample new particle states from the old ones stochastically using the "importance distribution", then reflect the change in certainty by adjusting the corresponding weight with a ratio between an evaluation of the new prediction on a posterior-like distribution (similar up to proportionality) divided by the evaluation of the prediction on the importance density. The reason we use this ratio is particular: Particles that move to places that better model the true posterior should receive more presence in the distribution, but we don't want to "oversample" regions that we draw from as the importance may not perfectly reflect the posterior. 

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

<img src="https://github.com/user-attachments/assets/f51a4f61-eae3-4796-a1d6-78f64bea318e" alt="image" width="400"/>

T = 500, resampling particles at N_s / 2, grid resolution 12px


This is a personal project made during Summer of 2025 developed by me, a rising sophomore studying computer science and mathematics at Rutgers University.

References
- Arulampalam, M. S., Maskell, S., Gordon, N., & Clapp, T. (2002). A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian tracking. IEEE Transactions on Signal Processing, 50(2), 174–188.
