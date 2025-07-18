module {
  func.func @main(%arg0: tensor<60x97x9x92xi1>, %arg1: tensor<16x25xi32>, %arg2: tensor<16x25xi32>, %arg3: tensor<24x42x17x72x78x60xf32>, %arg4: tensor<1x1x17x72x78x1xf32>) -> (tensor<60x97x9x1xi1>, tensor<16x25xi1>, tensor<16x25xi1>, tensor<24x42x17x72x78x60xf32>) {
    %0 = tosa.logical_not %arg0 : (tensor<60x97x9x92xi1>) -> tensor<60x97x9x92xi1>
    %1 = tosa.reduce_product %0 {axis = 3 : i32} : (tensor<60x97x9x92xi1>) -> tensor<60x97x9x1xi1>
    %2 = tosa.clz %1 : (tensor<60x97x9x1xi1>) -> tensor<60x97x9x1xi1>
    %3 = tosa.maximum %arg1, %arg2 : (tensor<16x25xi32>, tensor<16x25xi32>) -> tensor<16x25xi32>
    %4 = tosa.equal %3, %3 : (tensor<16x25xi32>, tensor<16x25xi32>) -> tensor<16x25xi1>
    %5 = tosa.logical_or %4, %4 : (tensor<16x25xi1>, tensor<16x25xi1>) -> tensor<16x25xi1>
    %6 = tosa.bitwise_not %3 : (tensor<16x25xi32>) -> tensor<16x25xi32>
    %7 = tosa.bitwise_xor %6, %6 : (tensor<16x25xi32>, tensor<16x25xi32>) -> tensor<16x25xi32>
    %8 = tosa.greater %7, %7 : (tensor<16x25xi32>, tensor<16x25xi32>) -> tensor<16x25xi1>
    %9 = tosa.pow %arg3, %arg4 : (tensor<24x42x17x72x78x60xf32>, tensor<1x1x17x72x78x1xf32>) -> tensor<24x42x17x72x78x60xf32>
    return %2, %5, %8, %9 : tensor<60x97x9x1xi1>, tensor<16x25xi1>, tensor<16x25xi1>, tensor<24x42x17x72x78x60xf32>
  }
}
