module {
  func.func @main(%arg0: tensor<7x19x73x75x81x17xi1>, %arg1: tensor<1x19x1x75x1x1xi1>, %arg2: tensor<22x83x21xi1>, %arg3: tensor<75x27x36x72x53xf32>) -> (tensor<7x19x73x75x81x17xi1>, tensor<66x332x126xi1>, tensor<75x27x36x72x53xf32>, tensor<1x166x1xi1>, tensor<75x27x36x72x53xf32>, tensor<1x332x1xi1>, tensor<1x166x63xi1>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<7x19x73x75x81x17xi1>, tensor<1x19x1x75x1x1xi1>) -> tensor<7x19x73x75x81x17xi1>
    %t_1 = tosa.const_shape {values = dense<[ 1, 2, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.tile %arg2, %t_1 : (tensor<22x83x21xi1>, !tosa.shape<3>) -> tensor<22x166x63xi1>
    %t_2 = tosa.const_shape {values = dense<[ 3, 2, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.tile %1, %t_2 : (tensor<22x166x63xi1>, !tosa.shape<3>) -> tensor<66x332x126xi1>
    %3 = tosa.exp %arg3 : (tensor<75x27x36x72x53xf32>) -> tensor<75x27x36x72x53xf32>
    %4 = tosa.maximum %3, %3 : (tensor<75x27x36x72x53xf32>, tensor<75x27x36x72x53xf32>) -> tensor<75x27x36x72x53xf32>
    %5 = tosa.reduce_max %1 {axis = 0 : i32} : (tensor<22x166x63xi1>) -> tensor<1x166x63xi1>
    %6 = tosa.reduce_product %5 {axis = 2 : i32} : (tensor<1x166x63xi1>) -> tensor<1x166x1xi1>
    %7 = tosa.reduce_product %1 {axis = 0 : i32} : (tensor<22x166x63xi1>) -> tensor<1x166x63xi1>
    %8 = tosa.ceil %3 : (tensor<75x27x36x72x53xf32>) -> tensor<75x27x36x72x53xf32>
    %9 = tosa.concat %7, %5 {axis = 1 : i32} : (tensor<1x166x63xi1>, tensor<1x166x63xi1>) -> tensor<1x332x63xi1>
    %10 = tosa.reduce_all %9 {axis = 2 : i32} : (tensor<1x332x63xi1>) -> tensor<1x332x1xi1>
    %11 = tosa.logical_left_shift %7, %5 : (tensor<1x166x63xi1>, tensor<1x166x63xi1>) -> tensor<1x166x63xi1>
    return %0, %2, %4, %6, %8, %10, %11 : tensor<7x19x73x75x81x17xi1>, tensor<66x332x126xi1>, tensor<75x27x36x72x53xf32>, tensor<1x166x1xi1>, tensor<75x27x36x72x53xf32>, tensor<1x332x1xi1>, tensor<1x166x63xi1>
  }
}
