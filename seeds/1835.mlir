module {
  func.func @main(%arg0: tensor<42x17x7x52x13x52xi32>, %arg1: tensor<71x17x7x52x13x52xi32>, %arg2: tensor<86xi32>, %arg3: tensor<i1>, %arg4: tensor<i1>, %arg5: tensor<97xi1>, %arg6: tensor<18xf32>) -> (tensor<113x17x7x52x13x52xi32>, tensor<i1>, tensor<172xi1>, tensor<18xf32>, tensor<172xi1>, tensor<1xi1>, tensor<1xi32>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<42x17x7x52x13x52xi32>, tensor<71x17x7x52x13x52xi32>) -> tensor<113x17x7x52x13x52xi32>
    %1 = tosa.intdiv %0, %0 : (tensor<113x17x7x52x13x52xi32>, tensor<113x17x7x52x13x52xi32>) -> tensor<113x17x7x52x13x52xi32>
    %t_2 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.tile %arg2, %t_2 : (tensor<86xi32>, !tosa.shape<1>) -> tensor<172xi32>
    %3 = tosa.logical_xor %arg3, %arg4 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = tosa.bitwise_or %2, %2 : (tensor<172xi32>, tensor<172xi32>) -> tensor<172xi32>
    %5 = tosa.reduce_all %arg5 {axis = 0 : i32} : (tensor<97xi1>) -> tensor<1xi1>
    %6 = tosa.logical_right_shift %2, %4 : (tensor<172xi32>, tensor<172xi32>) -> tensor<172xi32>
    %7 = tosa.greater_equal %6, %6 : (tensor<172xi32>, tensor<172xi32>) -> tensor<172xi1>
    %8 = tosa.greater_equal %6, %2 : (tensor<172xi32>, tensor<172xi32>) -> tensor<172xi1>
    %9 = tosa.reciprocal %arg6 : (tensor<18xf32>) -> tensor<18xf32>
    %10 = tosa.reduce_sum %6 {axis = 0 : i32} : (tensor<172xi32>) -> tensor<1xi32>
    %11 = tosa.reverse %10 {axis = 0 : i32} : (tensor<1xi32>) -> tensor<1xi32>
    %12 = tosa.bitwise_not %8 : (tensor<172xi1>) -> tensor<172xi1>
    %13 = tosa.add %5, %5 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %14 = tosa.reduce_sum %11 {axis = 0 : i32} : (tensor<1xi32>) -> tensor<1xi32>
    return %1, %3, %7, %9, %12, %13, %14 : tensor<113x17x7x52x13x52xi32>, tensor<i1>, tensor<172xi1>, tensor<18xf32>, tensor<172xi1>, tensor<1xi1>, tensor<1xi32>
  }
}
