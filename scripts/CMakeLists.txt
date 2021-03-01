project(mjx_scripts)

file(GLOB SCRIPTS_FILES *.cpp)
add_executable(${PROJECT_NAME} ${SCRIPTS_FILES})

target_link_libraries(${PROJECT_NAME} mjx)
