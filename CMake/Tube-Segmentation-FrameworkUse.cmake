###########################################################
##          Tube-Segmentation-Framework Use file
###########################################################

#------------------------------------------------------------------------------
# External libraries
#------------------------------------------------------------------------------
#set(CMAKE_MODULE_PATH
#    ${CMAKE_MODULE_PATH}
#    ${Tube-Segmentation-Framework_MODULE_PATH}
#    )
#include (tsfInitializeLibraries)

#------------------------------------------------------------------------------
# Where to look for includes and libraries
#------------------------------------------------------------------------------
include_directories( ${Tube-Segmentation-Framework_INCLUDE_DIRS}  ${Tube-Segmentation-Framework_BINARY_DIR})
link_directories (${Tube-Segmentation-Framework_LIBRARY_DIRS})
