cmake_minimum_required(VERSION 3.20)
project(Deeds C CXX)

# Additional CMake sources
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Configure output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optionally, use conan for package management
option(USE_CONAN "Use conan for dependency management" OFF)
if(USE_CONAN)
	include(conan)
	CONAN_INSTALL_DEPENDENCIES(CONANFILE ${CMAKE_CURRENT_LIST_DIR}/conanfile.txt)
endif()
add_compile_definitions(NOMINMAX)

# Find external libraries
find_package(TBB CONFIG REQUIRED)
find_package(ZLIB CONFIG REQUIRED)
find_package(argparse CONFIG REQUIRED)

# Global includes
include_directories(src)

# Sources
add_subdirectory(src)

