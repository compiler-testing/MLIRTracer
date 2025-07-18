module {
  func.func @main(%arg0: tensor<41x3x78x31x4xi8>, %arg1: tensor<1x3x78x31x1xi8>, %arg2: tensor<22x35x67x58x31x3xf32>, %arg3: tensor<76xi16>) -> (tensor<41x3x78x31x4xi8>, tensor<22x35x67x58x31x3xf32>, tensor<76xi16>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<41x3x78x31x4xi8>, tensor<1x3x78x31x1xi8>) -> tensor<41x3x78x31x4xi8>
    %1 = tosa.reciprocal %arg2 : (tensor<22x35x67x58x31x3xf32>) -> tensor<22x35x67x58x31x3xf32>
    %2 = tosa.reverse %arg3 {axis = 0 : i32} : (tensor<76xi16>) -> tensor<76xi16>
    return %0, %1, %2 : tensor<41x3x78x31x4xi8>, tensor<22x35x67x58x31x3xf32>, tensor<76xi16>
  }
}
