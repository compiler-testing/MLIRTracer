module {
  func.func @main(%arg0: tensor<8x53x14xi32>, %arg1: tensor<8x53x1xi32>, %arg2: tensor<4x20x57xi1>) -> (tensor<1x1x10xi32>, tensor<1x20x57xi1>, tensor<1x1x57xi1>, tensor<1x6x10xi32>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<8x53x14xi32>, tensor<8x53x1xi32>) -> tensor<8x53x14xi32>
    %s_1_start = tosa.const_shape {values = dense<[ 0, 1, 4 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_1_size = tosa.const_shape {values = dense<[ 12, 6, 10 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<8x53x14xi32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<12x6x10xi32>
    %2 = tosa.reduce_sum %1 {axis = 0 : i32} : (tensor<12x6x10xi32>) -> tensor<1x6x10xi32>
    %3 = tosa.reduce_min %2 {axis = 1 : i32} : (tensor<1x6x10xi32>) -> tensor<1x1x10xi32>
    %4 = tosa.bitwise_xor %3, %3 : (tensor<1x1x10xi32>, tensor<1x1x10xi32>) -> tensor<1x1x10xi32>
    %5 = tosa.add %4, %4 : (tensor<1x1x10xi32>, tensor<1x1x10xi32>) -> tensor<1x1x10xi32>
    %6 = tosa.reduce_any %arg2 {axis = 0 : i32} : (tensor<4x20x57xi1>) -> tensor<1x20x57xi1>
    %7 = tosa.logical_or %6, %6 : (tensor<1x20x57xi1>, tensor<1x20x57xi1>) -> tensor<1x20x57xi1>
    %8 = tosa.reduce_product %6 {axis = 1 : i32} : (tensor<1x20x57xi1>) -> tensor<1x1x57xi1>
    %9 = tosa.minimum %2, %2 : (tensor<1x6x10xi32>, tensor<1x6x10xi32>) -> tensor<1x6x10xi32>
    return %5, %7, %8, %9 : tensor<1x1x10xi32>, tensor<1x20x57xi1>, tensor<1x1x57xi1>, tensor<1x6x10xi32>
  }
}
