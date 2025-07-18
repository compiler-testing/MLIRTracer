module {
  func.func @main(%arg0: tensor<8xi32>, %arg1: tensor<8xf32>) -> (tensor<i32>, tensor<1xi1>, tensor<1xf32>, tensor<1xi1>) {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<8xi32>) -> tensor<i32>
    %1 = tosa.reduce_sum %arg1 {axis = 0 : i32} : (tensor<8xf32>) -> tensor<1xf32>
    %2 = tosa.reduce_sum %1 {axis = 0 : i32} : (tensor<1xf32>) -> tensor<1xf32>
    %3 = tosa.minimum %2, %2 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %4 = tosa.abs %3 : (tensor<1xf32>) -> tensor<1xf32>
    %5 = tosa.greater %1, %1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %6 = tosa.logical_not %5 : (tensor<1xi1>) -> tensor<1xi1>
    %7 = tosa.intdiv %0, %0 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %t_8 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %8 = tosa.tile %5, %t_8 : (tensor<1xi1>, !tosa.shape<1>) -> tensor<2xi1>
    %9 = tosa.logical_left_shift %8, %8 : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
    %10 = tosa.logical_xor %9, %8 : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
    %11 = tosa.logical_right_shift %6, %6 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %12 = tosa.reverse %11 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %13 = tosa.reverse %12 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %14 = tosa.rsqrt %4 : (tensor<1xf32>) -> tensor<1xf32>
    %15 = tosa.reduce_sum %10 {axis = 0 : i32} : (tensor<2xi1>) -> tensor<1xi1>
    return %7, %13, %14, %15 : tensor<i32>, tensor<1xi1>, tensor<1xf32>, tensor<1xi1>
  }
}
