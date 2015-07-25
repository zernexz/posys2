#http://www.cmake.org/cmake/help/cmake2.6docs.html
cmake_minimum_required (VERSION 2.6)
project (POSYS)

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 


set( CMAKE_MODULE_PATH ${POSYS_SOURCE_DIR}/cmake/modules )

find_package( GLM REQUIRED )
find_package( GLFW3 REQUIRED )
find_package( Boost REQUIRED)
FIND_PACKAGE( OpenCV REQUIRED )
find_package (OpenGL REQUIRED)

include_directories(${Boost_INCLUDE_DIR})

set( POSYS_LIBS ${OpenCV_LIBS} )
list( APPEND POSYS_LIBS ${Boost_LIBRARIES} )
list( APPEND POSYS_LIBS ${OPENGL_LIBRARIES} )
list( APPEND POSYS_LIBS ${OPENGL_LIBRARIES} )
list( APPEND POSYS_LIBS ${GLFW3_LIBRARY} )


if( UNIX )
add_definitions( -Wall  -Wunused-function -Wsign-compare  -std=c++11 -O3 -pedantic -fopenmp -pthread)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-function -Wsign-compare  -Wall -std=c++11 -O3 -pedantic -fopenmp -pthread -Wunused-function -Wunused-variable")

  #add_definitions( -Wunused-function -Wsign-compare)

  find_package( OpenGL REQUIRED )
  list( APPEND POSYS_LIBS ${OPENGL_gl_LIBRARY} )

  find_package( Threads REQUIRED )
  list( APPEND POSYS_LIBS ${CMAKE_THREAD_LIBS_INIT} )

  find_package( X11 REQUIRED )
  list( APPEND POSYS_LIBS ${X11_Xrandr_LIB} ${X11_Xxf86vm_LIB} ${X11_Xi_LIB} -lX11 )

  find_library( RT_LIB rt )
  list( APPEND POSYS_LIBS ${RT_LIB} )
endif()


set( POSYS_SOURCES 
	camera.cpp memdump.cpp qt_opengl.cpp )


ADD_EXECUTABLE(posys ${POSYS_SOURCES})
TARGET_LINK_LIBRARIES(posys ${POSYS_LIBS} -DBOOST_UBLAS_NDEBUG -lboost_serialization -std=c++11 -pedantic -fopenmp -pthread)


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube4.avi ${CMAKE_CURRENT_BINARY_DIR}/cube4.avi COPYONLY)
