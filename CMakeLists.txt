cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(imgpro LANGUAGES CXX CUDA)
project(cuda_base CXX CUDA)

find_package(OpenCV REQUIRED )
include(CTest)

add_library(cuda_base 
    cudaHeaders.h
    cudaMain.h
    cudaMain.cu
    tools.cu
    tools.h
  )

# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to particles
# will also build with -std=c++11
target_compile_features(cuda_base PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties( cuda_base
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_50,code=sm_50")

if(BUILD_TESTING)

  add_executable(imgpro main.cpp)

  set_target_properties(imgpro PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(imgpro PRIVATE cuda_base ${OpenCV_LIBS})
endif()