cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0167 NEW)

project(merlin VERSION 0.1.0)

include(CTest)
enable_testing()

# Set the C++ standard
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

# Optionally specify Boost root if installed via Homebrew
set(BOOST_ROOT "/opt/homebrew/opt/boost")
set(BOOST_INCLUDEDIR "${BOOST_ROOT}/include")
set(BOOST_LIBRARYDIR "${BOOST_ROOT}/lib")

# Find Boost
find_package(Boost REQUIRED COMPONENTS program_options thread)
set(BOOST_USE_STATIC_LIBS ON)
include_directories(include ${Boost_INCLUDE_DIRS})

add_executable(merlin 
    src/main.cpp
    src/bte.cpp
    src/cte.cpp
    src/em.cpp
    src/gibbs.cpp
    src/graph.cpp
    src/ijgp.cpp
    src/jglp.cpp
    src/lbp.cpp
    src/merlin.cpp
    src/program_options.cpp
    src/util.cpp
    src/wmb.cpp
    include/algorithm.h
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

set(LINK_LIBS ${Boost_LIBRARIES})
link_directories(${Boost_LIBRARY_DIRS})
target_link_libraries(${PROJECT_NAME} ${LINK_LIBS} ${LINK_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
