# TOMMA: Trajectory Optimization for Multiple Model-based Agents

![multiagent](media/multiagent.gif)

Easy-to-use trajectory optimization for multiple agents using (potentially nonlinear) dynamics models.

## Install

This setup works well with a python virtual environment.

```
git clone git@github.com:mbpeterson70/casadi_trajectory_optimization.git
cd casadi_trajectory_optimization
pip install .
```

## Code

`MultiAgentOptimization` in `multi_agent_optimization.py` can be used with a dynamics object to optimize trajectories (minimum-time) and perform model predictive control (fixed-time). A library of constraints can be easily added with the use of the `MultiAgentOptimization` class including adding input and state constraints, adding objects, and specifying the minimum distance between agents. 

See the [examples](examples/) directory for examples of using the `MultiAgentOptimization` class with Dubins and quadrotor dynamics.

## See Also

[ROS Rover Trajectory Optimization](https://github.com/mbpeterson70/rover_trajectory_opt_ros) is a ROS wrapper for using the TOMMA code on ground robots.
