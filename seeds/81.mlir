module {
  func.func @main(%arg0: tensor<35x54x17xf32>, %arg1: tensor<12x48x3x58xi8>, %arg2: tensor<1x1x3x58xi8>) -> (tensor<1x54x1xf32>, tensor<1x48x3x58xi8>) {
    %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<35x54x17xf32>) -> tensor<1x54x17xf32>
    %1 = tosa.reverse %0 {axis = 1 : i32} : (tensor<1x54x17xf32>) -> tensor<1x54x17xf32>
    %2 = tosa.bitwise_xor %arg1, %arg2 : (tensor<12x48x3x58xi8>, tensor<1x1x3x58xi8>) -> tensor<12x48x3x58xi8>
    %3 = tosa.reduce_sum %1 {axis = 2 : i32} : (tensor<1x54x17xf32>) -> tensor<1x54x1xf32>
    %4 = tosa.reduce_max %2 {axis = 0 : i32} : (tensor<12x48x3x58xi8>) -> tensor<1x48x3x58xi8>
    return %3, %4 : tensor<1x54x1xf32>, tensor<1x48x3x58xi8>
  }
}
