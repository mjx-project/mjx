project(mj_dataset)

file(GLOB SOURCE_FILES *.cpp)
file(GLOB HEADER_FILES *.h)
add_library(${PROJECT_NAME} STATIC
        ${SOURCE_FILES} ${HEADER_FILES})

add_executable(${PROJECT_NAME}.out main.cpp)
target_link_libraries(${PROJECT_NAME}.out mj ${PROJECT_NAME})

add_definitions(-DRESOURCES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources")
