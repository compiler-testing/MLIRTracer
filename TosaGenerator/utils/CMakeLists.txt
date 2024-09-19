add_mlir_library(FuzzerUtils
        create.cpp
        opinfo.cpp
        utils.cpp
        transfer.cpp
)

target_link_libraries(FuzzerUtils PRIVATE /usr/local/lib/libjsoncpp.a)