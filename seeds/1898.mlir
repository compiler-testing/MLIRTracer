module {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<92xf32>, %arg2: tensor<5x55x82xi1>) -> (tensor<1x1xi32>, tensor<92xf32>, tensor<5x55x1xi1>, tensor<10x55x1xi1>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<i32>) -> tensor<i32>
    %r_1 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.reshape %0, %r_1 : (tensor<i32>, !tosa.shape<2>) -> tensor<1x1xi32>
    %2 = tosa.reduce_max %1 {axis = 0 : i32} : (tensor<1x1xi32>) -> tensor<1x1xi32>
    %3 = tosa.tanh %arg1 : (tensor<92xf32>) -> tensor<92xf32>
    %4 = tosa.reduce_any %arg2 {axis = 2 : i32} : (tensor<5x55x82xi1>) -> tensor<5x55x1xi1>
    %5 = tosa.floor %3 : (tensor<92xf32>) -> tensor<92xf32>
    %6 = tosa.maximum %3, %5 : (tensor<92xf32>, tensor<92xf32>) -> tensor<92xf32>
    %7 = tosa.logical_xor %4, %4 : (tensor<5x55x1xi1>, tensor<5x55x1xi1>) -> tensor<5x55x1xi1>
    %8 = tosa.add %4, %7 : (tensor<5x55x1xi1>, tensor<5x55x1xi1>) -> tensor<5x55x1xi1>
    %9 = tosa.bitwise_or %8, %8 : (tensor<5x55x1xi1>, tensor<5x55x1xi1>) -> tensor<5x55x1xi1>
    %10 = tosa.concat %8, %4 {axis = 0 : i32} : (tensor<5x55x1xi1>, tensor<5x55x1xi1>) -> tensor<10x55x1xi1>
    return %2, %6, %9, %10 : tensor<1x1xi32>, tensor<92xf32>, tensor<5x55x1xi1>, tensor<10x55x1xi1>
  }
}
