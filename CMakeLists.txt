cmake_minimum_required(VERSION 3.10)
project(cat_dog_classifier)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add source files
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/layer.cpp
    src/neural_network.cpp
)

# Link OpenCV
target_link_libraries(${PROJECT_NAME} 
    ${OpenCV_LIBS}
    stdc++fs  # Add filesystem library
)

# Optional: Print OpenCV version and libraries
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")