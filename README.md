# Legendre Moment-Based Function Approximation

This repository contains code for the paper:

**"Function Approximations Valid in Both Time and Frequency Domains Using Legendre Moments"**

## Purpose

This code demonstrates how to approximate a function by matching its Legendre moments using a least-squares approach. The method is valid in both time and frequency domains, as described in the paper.

## Why is this public?

The journal requested that the code be made public. Anyone is welcome to use, modify, or contribute to this code.

## What does the code do?

- Approximates a target function (e.g., a combination of sine and cosine) by matching its Legendre moments.
- Uses least-squares optimization to find the best-fit discrete function values.
- Plots and compares the original and approximated functions.
- Prints a comparison of the Legendre moments for both the original and approximated functions.

## How to run

1. Make sure you have Python 3 and the following packages installed:
   - numpy
   - scipy
   - matplotlib
2. Run the main script:
   ```bash
   python 2_main_legender_moments.py
   ```

## Contributions

Contributions are welcome! Please feel free to submit issues or pull requests.

---

Author: hamidddds
