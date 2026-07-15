# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Merlin is

Merlin is a standalone C++ solver for probabilistic inference over graphical models (Bayesian networks, Markov random fields). It supports the tasks PR (partition function / probability of evidence), MAR (posterior marginals), MAP, MMAP (marginal MAP), and EM (maximum-likelihood parameter learning for Bayes nets only). Inference algorithms include Loopy Belief Propagation (LBP), Iterative Join-Graph Propagation (IJGP), Join-Graph Linear Programming (JGLP), Weighted Mini-Bucket (WMB), Bucket-Tree Elimination (BTE), Clique-Tree Elimination (CTE), and Gibbs sampling. WMB/IJGP/JGLP are parameterized by an **i-bound** that trades accuracy for cost (i-bound ≥ treewidth gives exact inference).

Requires **boost** (linked against `boost_program_options`, and `boost_thread` via CMake). Built with CMake at C++11.

## Build & run

**CLI binary (CMake — the only build for the solver):**
```
mkdir build && cd build
cmake ..
cmake --build .        # produces build/merlin
```

**Python bindings** (`Makefile.pybind`, the sole remaining Makefile): requires `pybind11` installed in a conda env named `Pybind` (hardcoded in the Makefile). `make -f Makefile.pybind python_api` builds `merlin.*.so`, importable as `import merlin`. The pybind build first assembles a static `lib/libmerlin.a` from all sources *except* `src/merlin_pybind.cpp`, then links the binding against it. `make -f Makefile.pybind all` builds the CLI binary into `bin/merlin` instead. It is independent of the CMake build.

**Tests** (GoogleTest, in `tests/`). The root CMake build enables them by default (`MERLIN_BUILD_TESTS=ON`) and fetches GoogleTest via `FetchContent` on first configure (needs network once). To build and run:
```
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure          # all tests (~0.4s)
./build/tests/merlin_unit_tests --gtest_filter=Factor.*   # one suite directly
```
`tests/unit/` covers the header value classes (`variable`, `variable_set`, `factor`, `graph`, indexing helpers, `my_set`, etc.) and `graphical_model` UAI parsing; `tests/integration/` runs `Merlin::run()` end-to-end and checks the MAR/PR result against the committed `cancer.uai.MAR` golden and EM against a pinned snapshot in `tests/fixtures/`. The test targets recompile the solver `src/*.cpp` (minus `main.cpp`/`merlin_pybind.cpp`) directly. Configure with `-DMERLIN_BUILD_TESTS=OFF` for a plain binary build with no GoogleTest dependency. There is no linter.

Two latent library quirks are documented (not worked around) by the tests: the `MapType&` overload of `ind2sub` (`variable_set.h`) is declared `void` but `return`s a value (rejected by strict compilers), and `MER_ENUM`'s string constructor only prefix-matches the first value and values right after a comma — later values are stored with a leading space (see `tests/unit/test_enum.cpp`).

You can also verify changes by running the binary against instances in `data/`, e.g.:
```
./build/merlin --input-file data/pedigree1.uai --evidence-file data/pedigree1.evid \
         --task MAR --algorithm wmb --ibound 4 --iterations 10 --output-format json
```
Output is written to `<input>.<TASK>.out` (or a `--output-file`); the EM task writes a new model to `<input>.EM`. See README.md for the full CLI flag list and the UAI file formats (input/evidence/virtual-evidence/query/dataset/output) — these formats are the authoritative interface contract and the README documents them in detail.

## Architecture

**Two entry points, one engine.** The CLI (`src/main.cpp` → `src/program_options.cpp`) and the Python module (`src/merlin_pybind.cpp`) both drive the same `Merlin` facade class (`include/merlin.h`, `src/merlin.cpp`). `main.cpp` parses options into a struct and calls a long sequence of `eng.set_*(...)` setters followed by `eng.run()`; the pybind module exposes those same setters/getters as Python properties. **When you add a configurable parameter, it must be threaded through all of: the `Merlin` member + setter/getter, `program_options` (CLI), and `merlin_pybind.cpp` (Python), to stay consistent across both front ends.**

**Dual I/O modes.** `Merlin` reads models/evidence/query/dataset either from files or from in-memory strings, selected by `set_use_files(bool)`. Every input has both a `*_file` and a `*_string` variant (e.g. `read_model(const char*)` vs `read_model(std::string)`). The string path exists primarily for the Python/embedded use case.

**Algorithm class hierarchy.** `include/algorithm.h` defines the pure-virtual `algorithm` interface (in `namespace merlin`): `init()`, `run()`, `logZ()/logZub()/logZlb()`, `ub()/lb()`, `best_config()`, and `belief(...)`. Each concrete algorithm is a class that inherits **both** `graphical_model` and `algorithm` (e.g. `class wmb: public graphical_model, public algorithm`), paired as `include/<algo>.h` + `src/<algo>.cpp` for `wmb`, `ijgp`, `jglp`, `lbp`, `bte`, `cte`, `gibbs`. `Merlin::run()` dispatches on the selected task+algorithm, constructs the right algorithm object, feeds it the model and evidence, runs it, and formats the solution (UAI or JSON). EM lives in `em.h/em.cpp` and internally uses CTE for exact inference.

**Model representation.** `graphical_model` (`include/graphical_model.h`) is the base data structure — a collection of `factor`s over `variable`/`variable_set`s — and extends `graph` (`include/graph.h`). Supporting value-type headers: `factor.h`, `factor_graph.h`, `variable.h`, `variable_set.h`, `index.h`, `indexed_heap.h`, `set.h`, `vector.h`, `observation.h` (used for EM training examples). Most of these are header-only.

**Constants & enums.** All task/algorithm/format codes are `#define`s in `include/base.h` (`MERLIN_TASK_*` = 10/20/30/40/50, `MERLIN_ALGO_*` = 1000–1009, `MERLIN_INPUT_*`, `MERLIN_OUTPUT_UAI/JSON`). The Python `Task`/`Algorithm`/`InputFormat`/`OutputFormat` IntEnums in `merlin_pybind.cpp` are built by hand from these same values — **keep them in sync when editing `base.h`.** `include/enum.h` is a separate string-izable enum macro (`MER_ENUM`) used internally for parsing/printing, unrelated to the `base.h` codes.

## Conventions

- Library code lives in `namespace merlin`; the top-level `Merlin` facade and the `MERLIN_*` macros are global.
- Header/source pairs: interface in `include/`, implementation in `src/`; several utility classes are header-only.
- New source files must be added to the `add_executable(...)` list in `CMakeLists.txt` (and to `MERLIN_SOLVER_SRCS` in `tests/CMakeLists.txt` if they should be tested). `merlin_pybind.cpp` is deliberately excluded from the CMake target; `Makefile.pybind` globs `src/*.cpp` automatically for the Python binding.
- `AOBB`, `AOBF`, `RBFAOO` algorithm codes exist but are **not implemented**.
