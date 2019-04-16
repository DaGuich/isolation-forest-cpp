cmake_minimum_required(VERSION 3.0)
project(isoforest)

set(CMAKE_CXX_STANDARD 17)

add_library(isoforest isolation_forest.cxx isolation_forest.hxx)

add_executable(example main.cxx data.hxx)
target_link_libraries(example PUBLIC isoforest)
