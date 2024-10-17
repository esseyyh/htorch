cmake_minimum_required(VERSION 3.10)
project(htorch VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add include directory
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add source files
set(SOURCES
    src/tensor/tensor.cpp
    src/htorch.cpp
)

# Create shared library
add_library(htorch SHARED ${SOURCES})

# Set the public header property for proper installation
#set_target_properties(htorch PROPERTIES
#    PUBLIC_HEADER "include/int/int.h;include/float/float.h;include/operations.h"
#)

# Install library
install(TARGETS htorch
    LIBRARY DESTINATION lib
)
# Install headers while maintaining directory structure
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ 
        DESTINATION include/htorch 
        FILES_MATCHING PATTERN "*.h")
# Uninstall target
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
