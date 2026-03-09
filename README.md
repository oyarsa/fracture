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

## Dependencies

- **C++ compiler** with C++14 support (e.g. g++ 13+)
- **GNU Make** 4.3+
- **Eigen** (header-only linear algebra library, included as a git submodule in `third_party/Eigen`)
- **Python 3.12+** and **uv** (for running analysis/plotting scripts)
- **GraphViz** (optional, for rendering circuit graphs as PDFs)

## Building

Clone the repository with submodules:

```bash
git clone --recurse-submodules <repo-url>
cd fracture
```

If you already cloned without `--recurse-submodules`, initialize Eigen with:

```bash
git submodule update --init
```

Build the simulator:

```bash
make                    # Development build (with AddressSanitizer/UBSan)
make BUILD=release      # Release build (optimized, with hardening flags)
```

The compiled binary is placed at `out/fracture`.

## Running

### Single simulation

```bash
./out/fracture [L] [D] [G]
```

**Parameters:**

| Parameter | Type   | Default | Description                                              |
|-----------|--------|---------|----------------------------------------------------------|
| `L`       | int    | 14      | Lattice size                                             |
| `D`       | float  | 1.0     | Disorder parameter in [0, 1]                             |
| `G`       | char   | t       | Circuit geometry: `s` (square), `t` (tilted), `h` (hex)  |

**Example:**

```bash
./out/fracture 20 0.5 s    # 20x20 square lattice, disorder 0.5
```

The simulation outputs:
- A summary line to stdout (geometry, size, disorder, iterations, time, max voltage, seed)
- `output.csv` with V-I curve data (voltage, current pairs)
- `graph-begin.dot` and `graph-end.dot` with the initial and final circuit states in GraphViz format

### Batch simulations and plotting

The Python scripts use `uv` for dependency management. To run batch simulations:

```bash
uv run python run.py -g t -l 14 -d 0.2 0.4 0.6 0.8 1.0 --show
```

Other analysis scripts:

```bash
uv run python plot.py -f data -g t -l 14          # Plot existing CSV data
uv run python curve.py -s data -m disorder -g t    # Plot V-I curves varying disorder
uv run python fit.py -s data -m size -g t          # Power law fitting
```

## Project structure

```
fracture.cpp          Main C++ simulation source
Makefile              Build configuration
run.py                Batch simulation runner with plotting
plot.py               Plot individual datasets
curve.py              Plot V-I curves by varying size/disorder
fit.py                Power law fitting and visualization
third_party/Eigen/    Eigen linear algebra library (submodule)
```

## How it works

The simulator solves for node voltages using Kirchhoff's Current Law (KCL) at each step.
The KCL system is solved via Cholesky decomposition (LLT) using Eigen. Currents through
each edge are computed from Ohm's law. If any fuse carries current above its maximum
rating, it is removed from the circuit. The voltage is then incremented and the process
repeats until the circuit is fully disconnected.

Three circuit topologies are supported: square lattice, tilted (45-degree) lattice,
and hexagonal lattice. Fuse resistances and maximum currents are randomized based on
the disorder parameter.

## License

This project is licensed under the GPL v3. See [LICENSE](LICENSE) for details.
