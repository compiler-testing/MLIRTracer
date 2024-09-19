# Exclude tests from libMLIR.so
add_mlir_library(MLIRFuzzer
    MIX.cpp
    tosaGen.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/TosaGen
  ${PROJECT_SOURCE_DIR}/include/json

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTestDialect
  )

target_include_directories(MLIRFuzzer
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test
  )

#set(lib_DIR /usr/local/lib)
#link_directories(${lib_DIR})
# target_link_libraries(TosaGenerator PUBLIC /usr/local/lib/libjsoncpp.a FuzzerUtils)
target_link_libraries(MLIRFuzzer PUBLIC libjsoncpp.a FuzzerUtils)

