module {
  func.func @main(%arg0: tensor<68xi64>, %arg1: tensor<68xi64>, %arg2: tensor<94x56xf32>) -> (tensor<1xi64>, tensor<1x56xf32>, tensor<94x56xf32>, tensor<1x56xi1>, tensor<94x56xi1>) {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = false} : (tensor<68xi64>, tensor<68xi64>) -> tensor<68xi64>
    %1 = tosa.ceil %arg2 : (tensor<94x56xf32>) -> tensor<94x56xf32>
    %2 = tosa.reduce_sum %0 {axis = 0 : i32} : (tensor<68xi64>) -> tensor<1xi64>
    %t_3 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.tile %1, %t_3 : (tensor<94x56xf32>, !tosa.shape<2>) -> tensor<94x56xf32>
    %4 = tosa.exp %3 : (tensor<94x56xf32>) -> tensor<94x56xf32>
    %5 = tosa.reduce_product %4 {axis = 0 : i32} : (tensor<94x56xf32>) -> tensor<1x56xf32>
    %6 = tosa.abs %5 : (tensor<1x56xf32>) -> tensor<1x56xf32>
    %7 = tosa.floor %6 : (tensor<1x56xf32>) -> tensor<1x56xf32>
    %8 = tosa.greater_equal %3, %1 : (tensor<94x56xf32>, tensor<94x56xf32>) -> tensor<94x56xi1>
    %9 = tosa.sub %7, %6 : (tensor<1x56xf32>, tensor<1x56xf32>) -> tensor<1x56xf32>
    %10 = tosa.bitwise_or %8, %8 : (tensor<94x56xi1>, tensor<94x56xi1>) -> tensor<94x56xi1>
    %11 = tosa.maximum %1, %1 : (tensor<94x56xf32>, tensor<94x56xf32>) -> tensor<94x56xf32>
    %12 = tosa.greater %5, %5 : (tensor<1x56xf32>, tensor<1x56xf32>) -> tensor<1x56xi1>
    %13 = tosa.bitwise_xor %8, %10 : (tensor<94x56xi1>, tensor<94x56xi1>) -> tensor<94x56xi1>
    return %2, %9, %11, %12, %13 : tensor<1xi64>, tensor<1x56xf32>, tensor<94x56xf32>, tensor<1x56xi1>, tensor<94x56xi1>
  }
}
