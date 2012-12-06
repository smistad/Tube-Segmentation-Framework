include (CMakeImportBuildSettings)
cmake_import_build_settings (${TSF_BUILD_SETTINGS_FILE})

find_package(Boost REQUIRED)

find_package (PkgConfig REQUIRED)
pkg_check_modules (GTK2 REQUIRED gtk+-2.0 gthread-2.0)

include_directories( ${TSF_INCLUDE_DIRS}  ${TSF_BINARY_DIR} ${Boost_INCLUDE_DIRS} ${GTK2_INCLUDE_DIRS})
link_directories (${TSF_LIBRARY_DIRS})

