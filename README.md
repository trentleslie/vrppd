# VRPPD Solver with Python and Google Maps API
This project aims to develop a solver for the Vehicle Routing Problem with Pickup and Delivery (VRPPD) using Python. It makes use of various libraries, including pandas for data manipulation, googlemaps for distance calculations using the Google Maps API, and the vrpy package for solving the routing problem.

The problem is defined as follows: given a fleet of vehicles, a central depot, and several customers who require service for either pickup or delivery, the goal is to find the least-cost set of vehicle routes such that each customer's demand is fulfilled, and each vehicle's route starts and ends at the depot.

This repository contains scripts to import customer data from a CSV file, process the data, generate a distance matrix using the Google Maps API, and solve the routing problem. The solution provides the optimal routes for each vehicle in the fleet, minimizing the total travel distance.

It also includes a Streamlit-based web interface to provide a user-friendly way to input problem parameters and view the results. This can be seen at http://vrppd.trentleslie.com, but it will not run with out the correct code.
