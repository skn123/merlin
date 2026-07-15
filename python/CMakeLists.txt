# Merlin Python bindings (pybind11 via FetchContent).

include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

# Build the Python extension module from the pybind wrapper plus the shared
# solver sources (MERLIN_SOLVER_SRCS is defined in the root CMakeLists.txt).
# The target is named merlin_python to avoid clashing with the `merlin`
# executable target, but the produced module is named `merlin` so it can be
# imported as `import merlin`.
pybind11_add_module(merlin_python
    ${CMAKE_SOURCE_DIR}/src/merlin_pybind.cpp
    ${MERLIN_SOLVER_SRCS}
)

set_target_properties(merlin_python PROPERTIES OUTPUT_NAME merlin)
target_include_directories(merlin_python PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(merlin_python PRIVATE dl ${Boost_LIBRARIES})
