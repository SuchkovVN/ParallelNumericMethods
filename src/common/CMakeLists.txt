set(TARGET_NAME "common")

file(GLOB_RECURSE SRCS "*.[c|h]pp")

add_library(${TARGET_NAME} STATIC ${SRCS})
target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
