set(TARGET_NAME "lab_01")

add_executable(${TARGET_NAME} "src/main.cpp")
target_link_libraries(${TARGET_NAME} PRIVATE common OpenMP::OpenMP_C OpenMP::OpenMP_CXX)