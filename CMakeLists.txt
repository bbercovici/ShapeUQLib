# MIT License
# Copyright (c) 2017 Benjamin Bercovici

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

# @file   CMakeLists.txt
# @Author Benjamin Bercovici (bebe0705@colorado.edu)
# @date   January, 2019
# @brief  CMake listing enabling compilation and installation of the ShapeUQLib library


cmake_minimum_required(VERSION 3.10)

set(LIB_NAME "ShapeUQLib" CACHE STRING "Name of library to be created.")

project(${LIB_NAME})
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/source/cmake)

add_definitions(-Wall -O2 )
set(CMAKE_CXX_FLAGS "-std=c++14")

# Include directories
include_directories(include)

# Find Armadillo 
find_package(Armadillo REQUIRED )
include_directories(${ARMADILLO_INCLUDE_DIRS})

# Find RBK 
find_package(RigidBodyKinematics REQUIRED PATHS ${RBK_LOC})
include_directories(${RBK_INCLUDE_DIR} )

# Find OpenMP
if (NOT NO_OMP)
	find_package(OpenMP)
endif()

# Add source files in root directory
add_library(${LIB_NAME} SHARED
	source/Bezier.cpp
	source/ControlPoint.cpp
	source/Element.cpp
	source/Facet.cpp
	source/FrameGraph.cpp
	source/RefFrame.cpp
	source/ShapeModel.cpp
	source/ShapeModelBezier.cpp
	source/ShapeModelTri.cpp
	)

# Linking
set(library_dependencies
	${ARMADILLO_LIBRARIES}
	${Boost_LIBRARIES}
	${RBK_LIBRARY})


if (UNIX AND NOT APPLE )
	target_link_libraries(${LIB_NAME} ${library_dependencies})
elseif (OPENMP_FOUND AND NOT NO_OMP)
	target_link_libraries(${LIB_NAME} ${library_dependencies} OpenMP::OpenMP_CXX)
else()
	target_link_libraries(${LIB_NAME} ${library_dependencies})
endif()

file(GLOB includes include/*.hpp)

# Installing
# If ${BREW} is true, installing from homebrew formula
if(NOT BREW)
	install (TARGETS ${LIB_NAME} DESTINATION /usr/local/lib/)
	install (DIRECTORY ${PROJECT_SOURCE_DIR}/share/${LIB_NAME} DESTINATION /usr/local/share)
	install (DIRECTORY ${PROJECT_SOURCE_DIR}/include/ShapeUQLib DESTINATION /usr/local/include FILES_MATCHING PATTERN "*.hpp")
endif()
