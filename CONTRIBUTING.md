# Contributing to Merlin

Thanks for your interest in contributing! This document covers how to set up
a development environment, the coding conventions used in this codebase, and
how to submit changes.

## Getting Started

1. Fork the repository and clone your fork.
2. Build the solver and run the test suite (see below) to confirm your
   environment is set up correctly before making changes.
3. Create a branch for your change: `git checkout -b my-feature`.

## Building

Merlin is built with CMake (>= 3.14) and requires a C++17 compiler and the
Boost `program_options` and `thread` libraries. See the "Build" section of
[README.md](README.md) for full prerequisites and instructions. In short:

```
cmake -S . -B build
cmake --build build
./build/merlin --help
```

The Python bindings are built separately via `Makefile.pybind` (see the
"Building the Python API" section of the README) and are not part of the
CMake build.

## Running the Tests

The test suite (GoogleTest) is built by default and can be run with:

```
ctest --test-dir build --output-on-failure
```

To iterate on a single test binary or filter to one suite:

```
./build/tests/merlin_unit_tests --gtest_filter=Factor.*
./build/tests/merlin_integration_tests
```

Please run the full suite before submitting a pull request, and add tests for
any new behavior or bug fix:

- **Unit tests** (`tests/unit/`) target the header-only value classes
  (`variable`, `variable_set`, `factor`, `graph`, indexing helpers, etc.) and
  `graphical_model` parsing.
- **Integration tests** (`tests/integration/`) drive the `Merlin` facade
  end-to-end and check results against golden files. If you fix a bug that
  previously crashed or produced wrong output on a specific input, add a
  regression test that reproduces it (see
  `Integration.PaskinCTEMARDoesNotCrash` for an example pairing a fix with a
  regression test).
- New solver source files must be added to `CMakeLists.txt` (and to
  `MERLIN_SOLVER_SRCS` in `tests/CMakeLists.txt` if they should be covered by
  the test targets).

If you introduce a new source file, remember it also needs to be added
consistently across the CMake target, the test target, and (if it affects the
public `Merlin` API) `src/merlin_pybind.cpp` for the Python bindings.

## Coding Conventions

- Library code lives in `namespace merlin`; the top-level `Merlin` facade and
  the `MERLIN_*` constants (`include/base.h`) are global.
- Match the existing style in the file you're editing: tabs for indentation
  in most of `include/` and `src/`, Doxygen-style `///` comments on public
  methods.
- Prefer small, focused changes. Avoid unrelated reformatting in the same
  commit/PR as a functional change.
- When changing file formats or CLI behavior, check whether `README.md`
  documents them and update it accordingly (the UAI input/evidence/query/
  dataset/output formats are documented there in detail).

## Reporting Bugs

When filing an issue, please include:

- The exact command line (or `Merlin` API calls) used to reproduce the
  problem.
- The input file(s) involved, or a minimal example that reproduces it.
- What you expected to happen and what actually happened (including any
  crash output, stack trace, or sanitizer report).

If you can, a minimal reproducer under `examples/` plus a proposed regression
test is the fastest path to a fix being accepted.

## Submitting Changes

1. Make sure `ctest --test-dir build --output-on-failure` passes.
2. Keep commits focused — one logical change per commit, with a clear
   message describing *why*, not just *what*.
3. Open a pull request describing the motivation for the change and how you
   verified it (tests added/run, manual reproduction steps, etc.).

By contributing, you agree that your contributions will be licensed under the
project's [BSD 3-Clause license](LICENSE).

## Code of Conduct

This project follows a [Code of Conduct](CODE_OF_CONDUCT.md). By
participating, you are expected to uphold it.

## Contact

Questions can be directed to the maintainer, Radu Marinescu
(`radu.marinescu@gmail.com`).
