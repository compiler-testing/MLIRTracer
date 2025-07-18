module {
  func.func @main(%arg0: tensor<37xf32>, %arg1: tensor<i1>) -> (tensor<37xf32>, tensor<37xf32>, tensor<i1>, tensor<1xf32>, tensor<74xf32>, tensor<74xi1>) {
    %0 = tosa.ceil %arg0 : (tensor<37xf32>) -> tensor<37xf32>
    %1 = tosa.ceil %0 : (tensor<37xf32>) -> tensor<37xf32>
    %2 = tosa.reduce_min %1 {axis = 0 : i32} : (tensor<37xf32>) -> tensor<1xf32>
    %3 = tosa.logical_not %arg1 : (tensor<i1>) -> tensor<i1>
    %4 = tosa.argmax %2 {axis = 0 : i32} : (tensor<1xf32>) -> tensor<i32>
    %5 = tosa.equal %4, %4 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %6 = tosa.log %0 : (tensor<37xf32>) -> tensor<37xf32>
    %7 = tosa.tanh %0 : (tensor<37xf32>) -> tensor<37xf32>
    %8 = tosa.log %1 : (tensor<37xf32>) -> tensor<37xf32>
    %t_9 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %9 = tosa.tile %8, %t_9 : (tensor<37xf32>, !tosa.shape<1>) -> tensor<74xf32>
    %10 = tosa.abs %5 : (tensor<i1>) -> tensor<i1>
    %11 = tosa.arithmetic_right_shift %10, %3 {round = true} : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %12 = tosa.reduce_sum %0 {axis = 0 : i32} : (tensor<37xf32>) -> tensor<1xf32>
    %13 = tosa.sigmoid %9 : (tensor<74xf32>) -> tensor<74xf32>
    %14 = tosa.equal %9, %9 : (tensor<74xf32>, tensor<74xf32>) -> tensor<74xi1>
    return %6, %7, %11, %12, %13, %14 : tensor<37xf32>, tensor<37xf32>, tensor<i1>, tensor<1xf32>, tensor<74xf32>, tensor<74xi1>
  }
}
