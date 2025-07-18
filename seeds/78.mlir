module {
  func.func @main(%arg0: tensor<45x40x9x8x67x84xf32>, %arg1: tensor<45x1x1x8x67x84xf32>, %arg2: tensor<7xi8>, %arg3: tensor<7xi8>) -> (tensor<45x40x9x8x67x84xf32>, tensor<7xi8>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<45x40x9x8x67x84xf32>, tensor<45x1x1x8x67x84xf32>) -> tensor<45x40x9x8x67x84xf32>
    %1 = tosa.identity %0 : (tensor<45x40x9x8x67x84xf32>) -> tensor<45x40x9x8x67x84xf32>
    %2 = tosa.bitwise_xor %arg2, %arg3 : (tensor<7xi8>, tensor<7xi8>) -> tensor<7xi8>
    return %1, %2 : tensor<45x40x9x8x67x84xf32>, tensor<7xi8>
  }
}
