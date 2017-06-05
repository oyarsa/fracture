# fracture

## Simulation of a scalar fracture model based on Freitas (2007)

Simulates a fracture system using an electrical circuit, modeling
forces as electrical currents over fuses and the breakages as fuses
blowing. This allow for a similar study but in a much simpler computational
setting, as electrical currents and potential difference are scalar values
as opposed to forces and Hooke's law, which deal with vectors.

The simulation is based on the principle that our circuit is divided in sectors,
and a current comes from an source point and moves through the circuit sector by
sector, in which we check its transmition from the input wires on a contact to
its output wires, where the proportion of the output current on each wire is inverse
to its resistance. After distributing the current the wire is checked to see if the
current is over its maximum value, and if it is the wire is removed from circuit.
This process goes until there is no path connecting the source point to the destination
point, i.e. there's no current flowing in the circuit anymore.
