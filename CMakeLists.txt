cmake_minimum_required(VERSION 3.14)
project(merlin VERSION 1.7.0)

option(MERLIN_BUILD_TESTS "Build the Merlin test suite" ON)
option(MERLIN_BUILD_PYTHON "Build the Python bindings" OFF)
option(MERLIN_BUILD_DOCS "Build the Doxygen documentation" OFF)

include(CTest)
enable_testing()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use Boost's own BoostConfig.cmake (config mode) instead of the deprecated
# bundled FindBoost module (removed path; see policy CMP0167, CMake >= 3.30).
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()
find_package(Boost COMPONENTS program_options thread REQUIRED)

include_directories(include)

# Solver sources shared by the CLI executable, the test targets, and the
# Python bindings (everything under src/ except the CLI entry point main.cpp
# and the pybind module merlin_pybind.cpp).
set(MERLIN_SOLVER_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/aobb.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/braobb.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bte.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cte.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/em.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gibbs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ijgp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/jglp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lbp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/merlin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/program_options.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/wmb.cpp
)

add_executable(merlin
    src/main.cpp
    ${MERLIN_SOLVER_SRCS}
    include/algorithm.h
    include/ao_search.h
    include/aobb.h
    include/braobb.h
    include/base.h
    include/bte.h
    include/cte.h
    include/em.h
    include/enum.h
    include/factor_graph.h
    include/factor.h
    include/gibbs.h
    include/graph.h
    include/graphical_model.h
    include/ijgp.h
    include/index.h
    include/indexed_heap.h
    include/jglp.h
    include/lbp.h
    include/merlin.h
    include/observation.h
    include/program_options.h
    include/set.h
    include/util.h
    include/variable_set.h
    include/variable.h
    include/vector.h
    include/wmb.h    
)

target_link_libraries(${PROJECT_NAME} dl Boost::program_options Boost::thread)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

if(MERLIN_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(MERLIN_BUILD_PYTHON)
    add_subdirectory(python)
endif()

if(MERLIN_BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    # Not part of ALL: build on demand with `cmake --build build --target docs`.
    # Run from the source dir so the Doxyfile's relative INPUT/OUTPUT_DIRECTORY
    # paths (include, src, README.md -> docs/) resolve correctly.
    add_custom_target(docs
        COMMAND Doxygen::doxygen ${CMAKE_CURRENT_SOURCE_DIR}/merlin.doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
endif()
