module {
  func.func @main(%arg0: tensor<76x9x43x33xi32>, %arg1: tensor<76x1x43x33xi32>, %arg2: tensor<61x32x45x73xf32>) -> (tensor<76x9x43x33xi32>, tensor<122x64x90x73xi1>, tensor<122x1x1x73xi1>) {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = false} : (tensor<76x9x43x33xi32>, tensor<76x1x43x33xi32>) -> tensor<76x9x43x33xi32>
    %1 = tosa.clamp %0 {min_val = 48 : i32, max_val = 154 : i32} : (tensor<76x9x43x33xi32>) -> tensor<76x9x43x33xi32>
    %2 = tosa.floor %arg2 : (tensor<61x32x45x73xf32>) -> tensor<61x32x45x73xf32>
    %3 = tosa.greater_equal %2, %2 : (tensor<61x32x45x73xf32>, tensor<61x32x45x73xf32>) -> tensor<61x32x45x73xi1>
    %4 = tosa.reduce_all %3 {axis = 1 : i32} : (tensor<61x32x45x73xi1>) -> tensor<61x1x45x73xi1>
    %5 = tosa.bitwise_xor %3, %3 : (tensor<61x32x45x73xi1>, tensor<61x32x45x73xi1>) -> tensor<61x32x45x73xi1>
    %6 = tosa.maximum %0, %1 : (tensor<76x9x43x33xi32>, tensor<76x9x43x33xi32>) -> tensor<76x9x43x33xi32>
    %7 = tosa.concat %4, %4 {axis = 0 : i32} : (tensor<61x1x45x73xi1>, tensor<61x1x45x73xi1>) -> tensor<122x1x45x73xi1>
    %8 = tosa.reduce_product %7 {axis = 2 : i32} : (tensor<122x1x45x73xi1>) -> tensor<122x1x1x73xi1>
    %t_9 = tosa.const_shape {values = dense<[ 2, 2, 2, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %9 = tosa.tile %5, %t_9 : (tensor<61x32x45x73xi1>, !tosa.shape<4>) -> tensor<122x64x90x73xi1>
    %10 = tosa.logical_xor %8, %8 : (tensor<122x1x1x73xi1>, tensor<122x1x1x73xi1>) -> tensor<122x1x1x73xi1>
    return %6, %9, %10 : tensor<76x9x43x33xi32>, tensor<122x64x90x73xi1>, tensor<122x1x1x73xi1>
  }
}
