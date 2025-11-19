
# set
set(SOURCE)
set(OBJECT)
set(BINARY nndeploy_demo_yolo_rtsp)
set(DIRECTORY demo)
set(DEPEND_LIBRARY)
set(SYSTEM_LIBRARY)
set(THIRD_PARTY_LIBRARY)

# Enable CUDA for GPU drawing kernels
enable_language(CUDA)

# include
include_directories(${ROOT_PATH}/demo)
include_directories(${ROOT_PATH}/demo/yolo_rtsp)

# SOURCE
file(GLOB_RECURSE SOURCE
  "${ROOT_PATH}/demo/yolo_rtsp/*.h"
  "${ROOT_PATH}/demo/yolo_rtsp/*.cc"
)
file(GLOB DEMO_SOURCE
  "${ROOT_PATH}/demo/*.h"
  "${ROOT_PATH}/demo/*.cc"
)

# CUDA kernel source
file(GLOB CUDA_SOURCE
  "${ROOT_PATH}/demo/yolo_rtsp/*.cu"
)

set(SOURCE ${SOURCE} ${DEMO_SOURCE} ${CUDA_SOURCE})
set(SOURCE ${SOURCE} ${ROOT_PATH}/plugin/source/nndeploy/force_link.cc)

# OBJECT
# BINARY
add_executable(${BINARY} ${SOURCE} ${OBJECT})
if (APPLE)
  set_target_properties(${BINARY} PROPERTIES LINK_FLAGS "-Wl,-undefined,dynamic_lookup")
elseif (UNIX)
  set_target_properties(${BINARY} PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
elseif(WIN32)
  if(MSVC)
    # target_link_options(${BINARY} PRIVATE /WHOLEARCHIVE)
  elseif(MINGW)
    set_target_properties(${BINARY} PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
  endif()
endif()
# DIRECTORY
set_property(TARGET ${BINARY} PROPERTY FOLDER ${DIRECTORY})
# DEPEND_LIBRARY
list(APPEND DEPEND_LIBRARY ${NNDEPLOY_FRAMEWORK_BINARY})
list(APPEND DEPEND_LIBRARY ${NNDEPLOY_DEPEND_LIBRARY})
list(APPEND DEPEND_LIBRARY ${NNDEPLOY_DEMO_DEPEND_LIBRARY})
target_link_libraries(${BINARY} ${DEPEND_LIBRARY})
# SYSTEM_LIBRARY
list(APPEND SYSTEM_LIBRARY ${NNDEPLOY_SYSTEM_LIBRARY})
list(APPEND SYSTEM_LIBRARY ${NNDEPLOY_DEMO_SYSTEM_LIBRARY})
target_link_libraries(${BINARY} ${SYSTEM_LIBRARY})
# THIRD_PARTY_LIBRARY
list(APPEND THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY})
list(APPEND THIRD_PARTY_LIBRARY ${NNDEPLOY_DEMO_THIRD_PARTY_LIBRARY})
list(APPEND THIRD_PARTY_LIBRARY ${NNDEPLOY_PLUGIN_THIRD_PARTY_LIBRARY})
list(APPEND THIRD_PARTY_LIBRARY ${NNDEPLOY_PLUGIN_LIST})

# FFmpeg libraries for GPU hardware decode
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAV REQUIRED
    libavformat
    libavcodec
    libavutil
    libswscale
)
include_directories(${LIBAV_INCLUDE_DIRS})
list(APPEND THIRD_PARTY_LIBRARY ${LIBAV_LIBRARIES})

# CUDA libraries for GPU drawing
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
list(APPEND THIRD_PARTY_LIBRARY ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY})

# OpenGL, GLEW and GLFW for GPU-direct display
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Freetype REQUIRED)
find_package(glm REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS} ${FREETYPE_INCLUDE_DIRS})
list(APPEND THIRD_PARTY_LIBRARY ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} glfw ${FREETYPE_LIBRARIES})

# Set CUDA properties for the target
set_target_properties(${BINARY} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(${BINARY} ${THIRD_PARTY_LIBRARY})
# install
if(SYSTEM_Windows)
  install(TARGETS ${BINARY} RUNTIME DESTINATION ${NNDEPLOY_INSTALL_DEMO_PATH})
else()
  install(TARGETS ${BINARY} RUNTIME DESTINATION ${NNDEPLOY_INSTALL_DEMO_PATH})
endif()

# unset
unset(SOURCE)
unset(OBJECT)
unset(BINARY)
unset(DIRECTORY)
unset(DEPEND_LIBRARY)
unset(SYSTEM_LIBRARY)
unset(THIRD_PARTY_LIBRARY)
