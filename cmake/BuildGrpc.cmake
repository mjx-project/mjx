include(FetchContent)

set(CMAKE_BUILD_TYPE Release)
set(gRPC_BUILD_TESTS OFF)
# set(gRPC_SSL_PROVIDER package)
# set(GRPC_FETCHCONTENT ON)

set(FETCHCONTENT_BASE_DIR ${EXTERNALDIR}/_deps)

FetchContent_Declare(
        protobuf
        GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
        GIT_TAG        v3.13.0
)

FetchContent_MakeAvailable(protobuf)

find_package(Protobuf REQUIRED)

FetchContent_Declare(
        gRPC
        GIT_REPOSITORY https://github.com/grpc/grpc
        GIT_TAG        v1.28.0
)

set(FETCHCONTENT_QUIET OFF)

FetchContent_MakeAvailable(gRPC)
