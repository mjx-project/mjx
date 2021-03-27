include(CTest)
include(${CMAKEDIR}/DownloadProject/DownloadProject.cmake)

set(DOWNLOAD_PROJ_NAME googletest)
download_project(PROJ                ${DOWNLOAD_PROJ_NAME}
                 GIT_REPOSITORY      https://github.com/google/googletest.git
                 GIT_TAG             master
                 DOWNLOAD_DIR        ${EXTERNALDIR}/${DOWNLOAD_PROJ_NAME}-download
                 SOURCE_DIR          ${EXTERNALDIR}/${DOWNLOAD_PROJ_NAME}-src
                 BINARY_DIR          ${EXTERNALDIR}/${DOWNLOAD_PROJ_NAME}-build
                 UPDATE_DISCONNECTED 1
)

add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
