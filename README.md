## pept-forward-model

![](figures/one_scatter_diagram.png)

_Positron Emission Particle Tracking_ (PEPT) is a nuclear imaging technique for observing the movement of particles
in opaque systems.

This is a very simplistic 'forward' model for simulating data observed by a real world detector. 
It is simple to use and understand compared to the more sophisticated software available and models two main sources
of uncertainty in the problem:
- [Compton scattering](https://en.wikipedia.org/wiki/Compton_scattering) of the photons.
- The discrete size of the cells that make up the detector.

This is primarily research code, so expect bugs.
### Usage

First it is necessary to create the relevant objects.
```python
from model import StaticParticle, CylinderDetector

p = StaticParticle()
d = CylinderDetector()
```
This creates a cylindrical detector and a particle at the origin tagged with a radioactive tracer. 
The default parameters are suitable for most cases, however we can configure certain aspects when needed
```python
# Move particle closer to the detector and up
p.set_position_cylindrical(r=0.1, theta=0, z=0.1)

# Adjusts the size of the detector cells in cm
d.detectors_height = 0.01
d.detectors_width = 0.01
```
To generate Line of Response (LoR) data, there is a single function call
```
lors, scatters = p.simulate_emissions(detector=d, n_emissions=10**4)
```
It is encouraged to examine the source code for these classes to familiarise oneself with the inner workings and
model parameters that can be set.
Example code is available (see `experiments.ipynb`) for help plotting and analysing the emission data.


Note that the scripts in this repository are intended to be run as modules from the source root, e.g
```bash
python -m inversion/generate_plots.py
```

This software is largely _research code_ so expect bugs.
#### Installation
```
pip install -r requirements.txt
```