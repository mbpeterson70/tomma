# Casadi Trajectory Optimization

![multiagent](media/multiagent.gif)

## Install

This setup works well with a python virtual environment.

```
git clone git@github.com:mbpeterson70/casadi_trajectory_optimization.git
cd casadi_trajectory_optimization
pip install .
```

## Code

`MultiAgentPlanner` in `multi_agent_planner.py` can be used with a dynamics object to optimize trajectories (minimum-time) and perform model predictive control (fixed-time). A library of constraints can be easily added with the use of the `MultiAgentPlanner` class including adding input and state constraints, adding objects, and specifying the minimum distance between agents. See the examples directory for examples of using the `MultiAgentPlanner` class with Dubins and quadrotor dynamics.

## See Also

[ROS Rover Trajectory Optimization](https://github.com/mbpeterson70/rover_trajectory_opt_ros) is a ROS wrapper for using the casadi trajectory optimization code on ground robots.