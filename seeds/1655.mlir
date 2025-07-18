module {
  func.func @main(%arg0: tensor<100x58xi1>, %arg1: tensor<1x1xi1>, %arg2: tensor<12x31x60x53x48x17xf32>, %arg3: tensor<1x31x1x1x1x1xf32>) -> (tensor<12x31x60x53x48x17xf32>, tensor<1x58xi1>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<100x58xi1>, tensor<1x1xi1>) -> tensor<100x58xi1>
    %1 = tosa.add %0, %0 : (tensor<100x58xi1>, tensor<100x58xi1>) -> tensor<100x58xi1>
    %2 = tosa.maximum %arg2, %arg3 : (tensor<12x31x60x53x48x17xf32>, tensor<1x31x1x1x1x1xf32>) -> tensor<12x31x60x53x48x17xf32>
    %3 = tosa.reduce_max %1 {axis = 0 : i32} : (tensor<100x58xi1>) -> tensor<1x58xi1>
    %4 = tosa.bitwise_xor %3, %3 : (tensor<1x58xi1>, tensor<1x58xi1>) -> tensor<1x58xi1>
    %5 = tosa.bitwise_not %4 : (tensor<1x58xi1>) -> tensor<1x58xi1>
    return %2, %5 : tensor<12x31x60x53x48x17xf32>, tensor<1x58xi1>
  }
}
