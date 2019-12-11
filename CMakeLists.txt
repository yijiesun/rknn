cmake_minimum_required(VERSION 2.8)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_BUILD_TYPE Release)

if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
    set(CMAKE_C_COMPILER gcc)
    set(CMAKE_CXX_COMPILER g++)
else()
    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


include_directories(
    ${CMAKE_SOURCE_DIR}/3rdparty/opencv/include
    ${CMAKE_SOURCE_DIR}/rknn_api/include
)

set(link_libs  ${CMAKE_SOURCE_DIR}/3rdparty/opencv/lib64/libopencv_core.so
               ${CMAKE_SOURCE_DIR}/3rdparty/opencv/lib64/libopencv_highgui.so
               ${CMAKE_SOURCE_DIR}/3rdparty/opencv/lib64/libopencv_imgcodecs.so
               ${CMAKE_SOURCE_DIR}/3rdparty/opencv/lib64/libopencv_imgproc.so
               ${CMAKE_SOURCE_DIR}/rknn_api/lib64/librknn_api.so
               pthread
)

add_executable(rknn_mobilenet  rknn_mobilenet.cpp)
target_link_libraries(rknn_mobilenet ${link_libs})

add_executable(rknn_ssd  rknn_ssd.cpp  v4l2/v4l2.cpp screen/screen.cpp config.cpp)
target_link_libraries(rknn_ssd ${link_libs})

add_executable(sreen_show  screen_show.cpp v4l2/v4l2.cpp screen/screen.cpp config.cpp)
target_link_libraries(sreen_show ${link_libs})

#add_executable(mt_ssd  rknn_ssd_multi.cpp v4l2/v4l2.cpp screen/screen.cpp knn/knn.cpp config.cpp)
#target_link_libraries(mt_ssd ${link_libs})

add_executable(mt_ssd_syj  rknn_ssd_multi_syj.cpp v4l2/v4l2.cpp screen/screen.cpp knn/knn.cpp config.cpp)
target_link_libraries(mt_ssd_syj ${link_libs})