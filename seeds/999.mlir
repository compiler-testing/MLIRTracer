module {
  func.func @main(%arg0: tensor<2x66xi32>, %arg1: tensor<2x1xi32>, %arg2: tensor<77x96x67x15x16x1xf32>, %arg3: tensor<38xi1>) -> (tensor<1xi1>, tensor<77x96x67x15x16x1xi1>, tensor<77x96x67x15x16x1xf32>, tensor<2x66xi32>, tensor<2x66xi1>, tensor<1x2xi1>, tensor<1xi1>) {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<2x66xi32>, tensor<2x1xi32>) -> tensor<2x66xi32>
    %1 = tosa.ceil %arg2 : (tensor<77x96x67x15x16x1xf32>) -> tensor<77x96x67x15x16x1xf32>
    %2 = tosa.reduce_all %arg3 {axis = 0 : i32} : (tensor<38xi1>) -> tensor<1xi1>
    %3 = tosa.reverse %2 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %4 = tosa.arithmetic_right_shift %0, %0 {round = true} : (tensor<2x66xi32>, tensor<2x66xi32>) -> tensor<2x66xi32>
    %5 = tosa.tanh %1 : (tensor<77x96x67x15x16x1xf32>) -> tensor<77x96x67x15x16x1xf32>
    %6 = tosa.greater %4, %4 : (tensor<2x66xi32>, tensor<2x66xi32>) -> tensor<2x66xi1>
    %7 = tosa.intdiv %4, %0 : (tensor<2x66xi32>, tensor<2x66xi32>) -> tensor<2x66xi32>
    %8 = tosa.bitwise_or %2, %3 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %s_9_start = tosa.const_shape {values = dense<[ 1, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_9_size = tosa.const_shape {values = dense<[ 1, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %9 = tosa.slice %6, %s_9_start, %s_9_size : (tensor<2x66xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x2xi1>
    %10 = tosa.reverse %9 {axis = 0 : i32} : (tensor<1x2xi1>) -> tensor<1x2xi1>
    %11 = tosa.logical_and %2, %8 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %12 = tosa.maximum %7, %7 : (tensor<2x66xi32>, tensor<2x66xi32>) -> tensor<2x66xi32>
    %13 = tosa.add %2, %11 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %14 = tosa.greater %1, %1 : (tensor<77x96x67x15x16x1xf32>, tensor<77x96x67x15x16x1xf32>) -> tensor<77x96x67x15x16x1xi1>
    %15 = tosa.floor %5 : (tensor<77x96x67x15x16x1xf32>) -> tensor<77x96x67x15x16x1xf32>
    %16 = tosa.minimum %12, %0 : (tensor<2x66xi32>, tensor<2x66xi32>) -> tensor<2x66xi32>
    %17 = tosa.reverse %10 {axis = 0 : i32} : (tensor<1x2xi1>) -> tensor<1x2xi1>
    %18 = tosa.greater_equal %0, %0 : (tensor<2x66xi32>, tensor<2x66xi32>) -> tensor<2x66xi1>
    %19 = tosa.logical_not %17 : (tensor<1x2xi1>) -> tensor<1x2xi1>
    %20 = tosa.bitwise_or %19, %9 : (tensor<1x2xi1>, tensor<1x2xi1>) -> tensor<1x2xi1>
    %21 = tosa.bitwise_xor %20, %20 : (tensor<1x2xi1>, tensor<1x2xi1>) -> tensor<1x2xi1>
    %22 = tosa.bitwise_or %2, %11 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    return %13, %14, %15, %16, %18, %21, %22 : tensor<1xi1>, tensor<77x96x67x15x16x1xi1>, tensor<77x96x67x15x16x1xf32>, tensor<2x66xi32>, tensor<2x66xi1>, tensor<1x2xi1>, tensor<1xi1>
  }
}
