module {
  func.func @main(%arg0: tensor<76x54x26x73xi64>, %arg1: tensor<76x1x1x73xi64>, %arg2: tensor<47x63x14x25x79xi1>, %arg3: tensor<47x1x14x1x1xi1>, %arg4: tensor<24x14x42x35x54xf32>) -> (tensor<47x63x14x25x79xi1>, tensor<24x14x42x35x54xf32>, tensor<228x54x78x3xi1>) {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<76x54x26x73xi64>, tensor<76x1x1x73xi64>) -> tensor<76x54x26x73xi64>
    %1 = tosa.reduce_min %0 {axis = 3 : i32} : (tensor<76x54x26x73xi64>) -> tensor<76x54x26x1xi64>
    %2 = tosa.logical_xor %arg2, %arg3 : (tensor<47x63x14x25x79xi1>, tensor<47x1x14x1x1xi1>) -> tensor<47x63x14x25x79xi1>
    %3 = tosa.log %arg4 : (tensor<24x14x42x35x54xf32>) -> tensor<24x14x42x35x54xf32>
    %t_4 = tosa.const_shape {values = dense<[ 3, 1, 3, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %4 = tosa.tile %1, %t_4 : (tensor<76x54x26x1xi64>, !tosa.shape<4>) -> tensor<228x54x78x3xi64>
    %5 = tosa.reverse %4 {axis = 0 : i32} : (tensor<228x54x78x3xi64>) -> tensor<228x54x78x3xi64>
    %6 = tosa.greater %5, %4 : (tensor<228x54x78x3xi64>, tensor<228x54x78x3xi64>) -> tensor<228x54x78x3xi1>
    return %2, %3, %6 : tensor<47x63x14x25x79xi1>, tensor<24x14x42x35x54xf32>, tensor<228x54x78x3xi1>
  }
}
