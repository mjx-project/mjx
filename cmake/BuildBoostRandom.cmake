include(${CMAKEDIR}/DownloadProject/DownloadProject.cmake)

set(DOWNLOAD_PROJ_NAME boostrandom)
download_project(PROJ                ${DOWNLOAD_PROJ_NAME}
                 GIT_REPOSITORY      https://github.com/boostorg/random
                 GIT_TAG             boost-1.75.0
                 DOWNLOAD_DIR        ${EXTERNALDIR}/${DOWNLOAD_PROJ_NAME}-download
                 SOURCE_DIR          ${EXTERNALDIR}/${DOWNLOAD_PROJ_NAME}-src
                 BINARY_DIR          ${EXTERNALDIR}/${DOWNLOAD_PROJ_NAME}-build
                 UPDATE_DISCONNECTED 1
)

include_directories(${boostrandom_SOURCE_DIR}/include)
