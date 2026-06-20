# motip

[![License: MIT](https://img.shields.io/badge/license-MIT-2F6B4F)](LICENSE)


`motip` is a research-oriented Python library for studying tensor-network contraction,
index-layout transformations, and memory behavior under different permutation strategies.
The repository combines a compact tensor-network model, contraction-path utilities,
layout-planning algorithms, memory estimators, experiment scripts, and optional interfaces to
high-performance tensor transposition and contraction backends.

## Contents

- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
- [Native Backends](#native-backends)
- [Citation](#citation)
- [License](#license)


## Repository Layout

```text
.
+-- src/
|   +-- tensor.py
|   +-- tensor_network/
|   +-- operations/
|   +-- memory/
+-- test/
+-- scripts/
+-- data/
+-- libs/
|   +-- blis/
|   +-- hptt/
|   +-- tccg/
|   +-- tcl/
+-- .github/workflows/
+-- pyproject.toml
+-- uv.lock
+-- CITATION.cff
+-- README.md
```

The bundled `libs/` directory contains external high-performance computing projects used for
native tensor contraction and transposition experiments. Their licensing terms are retained in
their respective subdirectories.

## Installation

The project targets Python 3.12 or newer and uses `uv` for dependency management.

```bash
uv sync --frozen
```

For local development, run commands through `uv`:

```bash
uv run pytest
uv run ruff check src/ test/ scripts/
uv run mypy check src/ test/ scripts/ --config-file pyproject.toml
```

Before you can get started, you will also need to install the dependencies included in the `libs/`
folder. Refer to each depedency, respectively, to install them.

## Quick Start

The following example constructs a random tensor network, obtains a contraction path from
`cotengra`, evaluates a permutation strategy, and reports memory estimates.

```python
import cotengra as ctg

from operations.strategy.next_use_aware import NextUseAwarePermutationStrategy
from tensor_network.utils.random import generate_random_tn

network = generate_random_tn(
    num_tensors=8,
    average_rank=3,
    max_dim=8,
    num_output_indices=2,
    seed=7,
)

tree = ctg.array_contract_tree(
    inputs=network.input_indices,
    output=network.output_indices,
    shapes=network.shapes,
)

path = tree.get_path()
operations = NextUseAwarePermutationStrategy.find_optimal_permutation(network, path)

peak_memory = NextUseAwarePermutationStrategy.get_peak_memory(network, path)
total_memory = NextUseAwarePermutationStrategy.get_total_memory(network, path)

print(f"operations: {len(operations)}")
print(f"peak memory: {peak_memory}")
print(f"total memory movement: {total_memory}")
```

## Core Components

### Tensor Representation

`Tensor` is the base data structure for tensor metadata. Each tensor stores:

- `input_indices`: integer labels for tensor modes;
- `shape`: dimensions corresponding to the input indices;
- `array`: an optional NumPy array for executable numerical contraction.

The constructor validates that each index has a matching dimension.

### Tensor Networks

`TensorNetwork` stores a collection of tensors, network output indices, and an index-size
dictionary. It can be initialized from `Tensor` objects or from raw index, shape, and array
components. The class also exposes:

- `input_indices`, `shapes`, and `tensor_arrays` views;
- `arrays` for workflows requiring concrete NumPy data;
- `as_graph`, a NetworkX representation whose nodes are tensors and whose edges represent
  shared indices;
- an internal tensor pool used during contraction-state updates.

`TensorNetworkBuilder` provides a fluent interface over `cotengra.utils.rand_equation` for
generating random networks with controlled tensor count, average rank, output-index count,
hyper-index count, dimension range, seed, and optional random arrays.

### Operations

Operations inherit from the `TensorOperation` abstraction and return `TensorOperationResult`
objects. The operation layer includes:

- tensor permutation operations for changing index order;
- tensor contraction operations for combining two tensors;
- dummy operations used in testing and pipeline construction;
- utility conversion functions for wrapping tensors into operation results.

The helper `apply_operations_to_network` applies a complete operation schedule to a tensor
network by first permuting initial tensors and then following the contraction path through
contraction and result-permutation steps.

### Contraction Paths and Trees

`ContractionPath` is represented as a sequence of tensor-index pairs. For analysis that
requires intermediate states, `PersistentContractionPath` stores:

- the original contraction path;
- the initial tensor network;
- each tensor-network state after every contraction.

`ContractionTree` reconstructs a binary tree from either a raw path or a persistent path.
Leaves correspond to initial tensors and internal nodes correspond to contraction steps. This
tree structure supports strategy implementations that need to reason about parent use,
intermediate tensors, and future layout requirements.

## Native Backends

The repository includes optional integrations with native tensor-computation libraries:

- `libs/hptt`: High-Performance Tensor Transposition support;
- `libs/tccg`: Tensor Contraction Code Generator support;
- `libs/tcl`: Tensor Contraction Library sources and examples;
- `libs/blis`: BLIS linear algebra sources used by native contraction tooling.

The TCCG interface under `src/operations/contraction/tccg_interface/` contains code for
discovering, generating, compiling, and dynamically loading generated contraction kernels.
Runtime execution is handled through dynamically loaded shared objects and NumPy arrays in
Fortran order.

Native backend installation is optional for pure metadata and memory-model experiments, but it
is required for workflows that execute generated kernels or backend-specific transposition
paths.

## Citation

If you use this software in academic work, please cite it as software.

```bibtex
@software{shihata_motip_2026,
  author = {Fam Shihata},
  title = {motip: Memory-aware tensor-index permutation strategies for tensor-network contraction research},
  year = {2026},
  version = {0.1.0},
  license = {MIT},
  url = {https://github.com/FamALouiz/motip}
}
```

## License

`motip` is distributed under the MIT License. See [LICENSE](LICENSE) for the full license text.

Bundled third-party libraries under `libs/` retain their own licenses and attribution files.
