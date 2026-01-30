# XTC - Xdsl Transform Compiler

The XTC project, aims at providing a high level scheduling specifications over an operation data flow graph
and runtime tool to ease evaluation of kernel transformations with several compiler backends on different targets.

## About

The projet provides high-level syntax for applying transformations such as:

- Tiling
- Vectorization
- Unrolling
- Fusion

It provides a high level interface for defining a dataflow graph of predefined operator such as:

- Convolutions
- Matrix multiplication
- Elementwise operations

It allow the integration of several backend and implements:

- MLIR backend (using MLIR linalg and transform dialects) for CPU and GPU
- TVM backend (using TVM Tensor IR and TVM Schedule APIs) for CPU
- JIR INRIA internal backend (using JIR IR and MLIR scf and affine dialects) for CPU

It provides some runtime to execute on different targets such as x86_64/Linux, aarch64/Linux, arm64/Mac, NVidia GPU.

Here is an overview of core components:

- Abstract interfaces for tensors, graphs, nodes, and operators
- Schedulers for transformation management
- Compilers for code generation
- Evaluators for performance measurement
- Backends for implementations of the different interfaces
- Strategies for transformation design space definition
- Models for autotuning based on iterative optimization

Learn more with the [Overview](overview.md).

For details, get to the [Architecture Design](design/architecture.md) or [Reference API](reference/api.md).

## Links

Refer to the [Source Repository](https://github.com/xtc-tools/xtc).

Refer to the [Arxiv Paper](https://arxiv.org/abs/2512.16512).
