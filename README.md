Hello! This is a project I made that models various recursive filters in a discrete time approach (extended kalman filter, SIS filter, approximated grid-based approach) within a 2d sim.
The goal is the model the position of a dot on the screen who's true location is known with "uncertainty" (assuming a robotics application this 
would be stuff like noise in the sensor data, inaccuracies in the hardware, wheel slippage ...). 
The problem arises of estimating the state at a certain discrete time step
given the previous state (the properties of the system that we track) and new observations (sensors, external sources of observation). 
The goal of the recursive filter is at each time step to propagate the state at the previous time step through "system dynamics" (for example, velocity over
time equations) to arrive at an estimated distribution of the system state called the prior, before modifying the prior with the latest observation 
to attain a more promising prediction. This is the posterior of the system, given as a conditional distribution of the current system state given 
the set of all observations up to time step k. By the discrete time step nature of the problem, the observations and states are random processes
consisting of sets of random vectors indexed by time. The special property of these recursive systems is that we are recursively generating the random vector denoting the state at each time step k using only the previous state and a new observation vector - 
that is, we only need information of the previous time step and current data to arrive at the new result. Thus, it makes these algorithms a very efficient and elegant solution 
to the state estimation problem. 
In practice, the system can be anything like a moving line in the stock market or the properties of an economic system, 
but in this case it will be a moveable "dot" on the screen.
The dot on the screen has a relatively simple state space and control input. Namely, the properties of the system we wish to track are its x and y coordinate pixels 
as well as user controlled velocity. Our sensor is a beacon at the center that tracks distance to the dot, but with error. Furthermore, I have incorporated
error within the dot's movement to better simulate realistic conditions that do not always act perfectly. 

*Particle Filter*
The particle filter is a special type of recursive filter that models the posterior with discrete spikes called particles. Each 
particle has an importance weight, which is just a fancy term for the probability of the system having the particular state of the particle. 
We sample new particle states from an importance distribution and provide a correction to the existing weight with a ratio between evaluations
of the posterior and importance at the current density with the sampled particle and new sensor data as inputs. 

<img src="https://github.com/user-attachments/assets/017e958b-6f67-4230-9afa-0b6751cc9370" alt="image" width="400"/>

The particles are distributed uniformly across the screen. 

<img src="https://github.com/user-attachments/assets/8662d259-0744-4585-8086-7bfcdd2b841e" alt="image" width="400"/>

In an instant, the likelihood function adjust the weights of particles and the resampling process (I use systematic sampling) 
causes the particles to "respawn" around these high density regions. Notice that the particles form a circle around the beacon, 
expected behavior as the beacon only gives distance to the dot, but not the orientation of the distance. 

<img src="https://github.com/user-attachments/assets/e7d1fd58-81fd-44c2-a9bb-ef1e1138de4b" alt="image" width="400"/>

Finally, the beauty of the particle filter is shown in full effect when the user makes an input. 

<img src="https://github.com/user-attachments/assets/3126c4f3-ee85-48fe-bfeb-4ceee12edc17" alt="image" width="400"/>

The preceding examples resample at every time step, but we can choose to resample only after the effective sample size metric 
falls below a certain threshold. I visualized higher probability particles as being larger. Notice that this leads to some
degeneracy as many particles have a negligible impact on the pdf. 

<img src="https://github.com/user-attachments/assets/c146da04-b008-4c4a-99ad-b35fec934075" alt="image" width="400"/>

We can choose to increase the number of particles at the cost of computational resources. This is with 10000 particles. 
Within the constraints of this particular sim, ~150 particles seems to have the best computation-to-convergence-rate trade off,
but in other applications the particle count may be increased further to reduce error. 


This is a personal project made during Summer of 2025. 

References
- This project is based on the methodology described in Arulampalam, M. S., Maskell, S., Gordon, N., & Clapp, T. (2002). A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian tracking. IEEE Transactions on Signal Processing, 50(2), 174–188.
