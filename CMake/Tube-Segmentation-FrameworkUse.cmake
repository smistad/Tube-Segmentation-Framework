###########################################################
##          Tube-Segmentation-Framework Use file
###########################################################

#------------------------------------------------------------------------------
# Options
#------------------------------------------------------------------------------
option (TSF_USE_EXTRNAL_OUL "Use external OpenCLUtilityLibrary" OFF)

#------------------------------------------------------------------------------
# External libraries
#------------------------------------------------------------------------------
# Boost
find_package(Boost REQUIRED)

# SIPL
find_package(SIPL PATHS "${Tube-Segmentation-Framework_BINARY_DIR}/SIPL" REQUIRED)
include(${SIPL_USE_FILE})

# OpenCLUtilityLibrary
if(TSF_USE_EXTRNAL_OUL)
    message(STATUS "Using external use file for OpenCLUtilityLibrary in TSF: "${TSF_EXTERNAL_OUL_USEFILE})
    include(${TSF_EXTERNAL_OUL_USEFILE})
else(TSF_USE_EXTRNAL_OUL)
    message(STATUS "Using submodule for OpenCLUtility in TSF")
    find_package(OpenCLUtilityLibrary PATHS "${Tube-Segmentation-Framework_BINARY_DIR}/OpenCLUtilityLibrary" REQUIRED)
    include(${OCL-Utilities_USE_FILE})
endif(TSF_USE_EXTRNAL_OUL)

#------------------------------------------------------------------------------
# Where to look for includes and libraries
#------------------------------------------------------------------------------
include_directories( ${Tube-Segmentation-Framework_INCLUDE_DIRS}  ${Tube-Segmentation-Framework_BINARY_DIR} ${Boost_INCLUDE_DIRS})
link_directories (${Tube-Segmentation-Framework_LIBRARY_DIRS})
