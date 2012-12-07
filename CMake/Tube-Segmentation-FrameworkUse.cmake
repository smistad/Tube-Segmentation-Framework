include (CMakeImportBuildSettings)
cmake_import_build_settings (${Tube-Segmentation-Framework_BUILD_SETTINGS_FILE})

find_package(Boost REQUIRED)

find_package (PkgConfig REQUIRED)
pkg_check_modules (GTK2 REQUIRED gtk+-2.0 gthread-2.0)

include_directories( ${Tube-Segmentation-Framework_INCLUDE_DIRS}  ${Tube-Segmentation-Framework_BINARY_DIR} ${Boost_INCLUDE_DIRS} ${GTK2_INCLUDE_DIRS})
link_directories (${Tube-Segmentation-Framework_LIBRARY_DIRS})

