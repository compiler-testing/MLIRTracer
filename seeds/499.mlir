module {
  func.func @main(%arg0: tensor<8x67x16x94x1xi32>, %arg1: tensor<1x67x16x1x1xi32>, %arg2: tensor<85x25x35x73xf32>) -> (tensor<16x67x16x94x1xi32>, tensor<85x25x35x73xf32>, tensor<85x25x35x73xf32>) {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<8x67x16x94x1xi32>, tensor<1x67x16x1x1xi32>) -> tensor<8x67x16x94x1xi32>
    %1 = tosa.reciprocal %arg2 : (tensor<85x25x35x73xf32>) -> tensor<85x25x35x73xf32>
    %2 = tosa.bitwise_xor %0, %0 : (tensor<8x67x16x94x1xi32>, tensor<8x67x16x94x1xi32>) -> tensor<8x67x16x94x1xi32>
    %3 = tosa.concat %2, %0 {axis = 0 : i32} : (tensor<8x67x16x94x1xi32>, tensor<8x67x16x94x1xi32>) -> tensor<16x67x16x94x1xi32>
    %4 = tosa.reciprocal %1 : (tensor<85x25x35x73xf32>) -> tensor<85x25x35x73xf32>
    %5 = tosa.exp %1 : (tensor<85x25x35x73xf32>) -> tensor<85x25x35x73xf32>
    return %3, %4, %5 : tensor<16x67x16x94x1xi32>, tensor<85x25x35x73xf32>, tensor<85x25x35x73xf32>
  }
}
