###########################################################
##          Tube-Segmentation-Framework Use file
###########################################################

#------------------------------------------------------------------------------
# External libraries
#------------------------------------------------------------------------------
# Boost
find_package(Boost REQUIRED)

# GTK
#find_package (PkgConfig REQUIRED)
#pkg_check_modules (GTK2 REQUIRED gtk+-2.0 gthread-2.0)

# SIPL
find_package(SIPL PATHS "${Tube-Segmentation-Framework_BINARY_DIR}/SIPL" REQUIRED)
include(${SIPL_USE_FILE})

# OpenCLUtilities
find_package(OCL-Utilities PATHS "${Tube-Segmentation-Framework_BINARY_DIR}/OpenCLUtilities" REQUIRED)
include(${OCL-Utilities_USE_FILE})

#------------------------------------------------------------------------------
# Where to look for includes and libraries
#------------------------------------------------------------------------------
include_directories( ${Tube-Segmentation-Framework_INCLUDE_DIRS}  ${Tube-Segmentation-Framework_BINARY_DIR} ${Boost_INCLUDE_DIRS})
link_directories (${Tube-Segmentation-Framework_LIBRARY_DIRS})

