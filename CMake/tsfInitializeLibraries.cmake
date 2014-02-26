###############################################################################
# Initialize external libraries for TSF
#
#
###############################################################################
macro(tsf_initialize_external_libraries)
    message( STATUS "Initializing External Libraries" )
    tsf_initialize_oul()
    tsf_initialize_boost()
    add_subdirectory(SIPL)
    tsf_initialize_sipl()
    tsf_initialize_openmp()
endmacro(tsf_initialize_external_libraries)


###############################################################################
# Initialize OpenCLUtility library
#
# Uses predefined variables:
#    TSF_USE_EXTRNAL_OUL : path to oul
#    TSF_EXTERNAL_OUL_PATH : path to oul build dir
#
###############################################################################
macro(tsf_initialize_oul)
    if(TSF_USE_EXTRNAL_OUL)
        message(STATUS "Using external OpenCLUtilityLibrary in TSF.")
        find_package(OpenCLUtilityLibrary PATHS ${TSF_EXTERNAL_OUL_PATH})
        message(STATUS "OpenCLUtilityLibrary_USE_FILE "${OpenCLUtilityLibrary_USE_FILE})
        include(${OpenCLUtilityLibrary_USE_FILE})
    else(TSF_USE_EXTRNAL_OUL)
        message(STATUS "Using submodule for OpenCLUtility in TSF")
        add_subdirectory(OpenCLUtilityLibrary)
        find_package(OpenCLUtilityLibrary PATHS "${Tube-Segmentation-Framework_BINARY_DIR}/OpenCLUtilityLibrary" REQUIRED)
        include(${OpenCLUtilityLibrary_USE_FILE})
    endif(TSF_USE_EXTRNAL_OUL)
endmacro(tsf_initialize_oul)


###############################################################################
# Initialize Boost library
#
###############################################################################
macro(tsf_initialize_boost)
    find_package(Boost COMPONENTS iostreams REQUIRED)
endmacro(tsf_initialize_boost)

###############################################################################
# Initialize SIPL library
#
###############################################################################
macro(tsf_initialize_sipl)
    find_package(SIPL PATHS "${Tube-Segmentation-Framework_BINARY_DIR}/SIPL" REQUIRED)
    include(${SIPL_USE_FILE})
endmacro(tsf_initialize_sipl)

###############################################################################
# Initialize OpenMP library
#
###############################################################################
macro(tsf_initialize_openmp)
    find_package(OpenMP)
    if(OPENMP_FOUND)
        message("-- OpenMP was detected. Using OpenMP to speed up some calculations.")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}" )
    endif()
endmacro(tsf_initialize_openmp)

###############################################################################
# Initialize GTest library
#
###############################################################################
macro(tsf_enable_testing)
    find_package(GTest)
    if(GTEST_FOUND AND SIPL_USE_GTK)
    	message("Google test framework found. Enabling testing ...")
    	target_link_libraries(tubeSegmentationLib OpenCLUtilityLibrary SIPL ${Boost_LIBRARIES})
    	enable_testing()
    	add_subdirectory(tests)
    endif()
endmacro(tsf_enable_testing)
