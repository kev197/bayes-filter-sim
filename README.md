Hello! This is a project I made that models various recursive filters in a discrete time approach (extended kalman filter, SIS filter, approximated grid-based approach) within a 2d sim.
The goal is the model the position of a dot on the screen who's true location is known with "uncertainty" (assuming a robotics application this 
would be stuff like noise in the sensor data, inaccuracies in the hardware, wheel slippage ...). 
The problem arises of estimating the state at a certain discrete time step
given the previous state (the properties of the system that we track) and new observations (sensors, external sources of observation). 
The goal of the recursive filter is at each time step to propagate the state at the previous time step through "system dynamics" (for example, velocity over
time equations) to arrive at an estimated distribution of the system state called the prior, before modifying the prior with the latest observation 
to attain a more promising prediction. This is the posterior of the system, given as a conditional distribution of the current system state given 
the set of all observations up to time step k. By the discrete time step nature of the problem, the observation and states are random processes
consisting of sets of random vectors indexed by time. The special property of these recursive systems is that we are recursively updating our prediction - 
that is, we only need information of the previous time step to arrive at the new result. Thus, it makes these algorithms a very efficient and elegant solution 
to the state estimation problem. 
In pracitce, the system can be anything like a moving line in the stock market or the properties of an economic system, 
but in this case it will be a moveable "dot" on the screen.
The dot on the screen has a relatively simple state space and control input. Namely, the properties of the system we wish to track are its x and y coordinate pixels 
as well as user controlled velocity. Our sensor is a beacon at the center that tracks distance to the dot, but with error. Furthermore, I have incorporated
error within the dot's movement to better simulate realistic conditions that do not always act perfectly. 

This is a personal project made during Summer of 2025. 

References
- This project is based on the methodology described in Arulampalam, M. S., Maskell, S., Gordon, N., & Clapp, T. (2002). A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian tracking. IEEE Transactions on Signal Processing, 50(2), 174–188.
