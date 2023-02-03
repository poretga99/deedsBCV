#-----------------------------------------------------------------------------
option(USE_NII_NAN "To enable NaN support with nifti images" ON)
mark_as_advanced(FORCE USE_NII_NAN)
if(USE_NII_NAN)
    add_definitions(-DUSE_NII_NAN)
endif(USE_NII_NAN)
add_definitions(-DHAVE_ZLIB)
add_library(nifti STATIC nifti1.h nifti1_io.c nifti1_io.h znzlib.c znzlib.h)
add_library(nifti::nifti ALIAS nifti)
target_include_directories(nifti PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(nifti PUBLIC ${ZLIB_LIBRARIES})
target_compile_definitions(nifti PRIVATE "_CRT_SECURE_NO_WARNINGS")
#-----------------------------------------------------------------------------
