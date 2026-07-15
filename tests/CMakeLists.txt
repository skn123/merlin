# Merlin test suite (GoogleTest via FetchContent).

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
)
# Do not override the parent project's compiler/linker settings on Windows.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# The solver sources compiled into each test target come from the shared
# MERLIN_SOLVER_SRCS list defined in the root CMakeLists.txt (visible here via
# add_subdirectory). It excludes main.cpp and the pybind module.

set(MERLIN_TEST_LIBS
    GTest::gtest_main
    dl
    Boost::program_options
    Boost::thread
)

# ----- Unit tests -----
add_executable(merlin_unit_tests
    ${MERLIN_SOLVER_SRCS}
    unit/test_variable.cpp
    unit/test_variable_set.cpp
    unit/test_my_vector_set.cpp
    unit/test_factor.cpp
    unit/test_index.cpp
    unit/test_graph.cpp
    unit/test_indexed_heap.cpp
    unit/test_observation.cpp
    unit/test_enum.cpp
    unit/test_graphical_model.cpp
)
target_include_directories(merlin_unit_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(merlin_unit_tests PRIVATE
    MERLIN_DATA_DIR="${CMAKE_SOURCE_DIR}/examples"
    MERLIN_ROOT_DIR="${CMAKE_SOURCE_DIR}"
)
target_link_libraries(merlin_unit_tests PRIVATE ${MERLIN_TEST_LIBS})
gtest_discover_tests(merlin_unit_tests)

# ----- Integration tests (end-to-end inference vs committed goldens) -----
add_executable(merlin_integration_tests
    ${MERLIN_SOLVER_SRCS}
    integration/test_inference_goldens.cpp
)
target_include_directories(merlin_integration_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(merlin_integration_tests PRIVATE
    MERLIN_DATA_DIR="${CMAKE_SOURCE_DIR}/examples"
    MERLIN_ROOT_DIR="${CMAKE_SOURCE_DIR}"
    MERLIN_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
    MERLIN_TEST_TMP_DIR="${CMAKE_CURRENT_BINARY_DIR}"
)
target_link_libraries(merlin_integration_tests PRIVATE ${MERLIN_TEST_LIBS})
gtest_discover_tests(merlin_integration_tests)
