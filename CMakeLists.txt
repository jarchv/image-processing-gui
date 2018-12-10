cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(imgpro LANGUAGES CXX CUDA)
project(cuda_base CXX CUDA)

find_package(OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(include)


include(CTest)
#include(${wxWidgets_USE_FILE})
#include(FindCUDA)

add_library(cuda_base 
    src/cudaHeaders.h
    src/cudaMain.h
    src/cudaMain.cu
    src/tools.cu
    src/tools.h
  )

target_compile_features(cuda_base PUBLIC cxx_std_11)

set_target_properties( cuda_base
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_50,code=sm_50")

if(BUILD_TESTING)

  add_executable(imgpro src/main.cpp)

  set_target_properties(imgpro PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(imgpro PRIVATE cuda_base ${OpenCV_LIBS})# ${wxWidgets_LIBRARIES})
  install(TARGETS imgpro DESTINATION bin)
endif()
