module {
  func.func @main(%arg0: tensor<18xi64>, %arg1: tensor<18xi64>, %arg2: tensor<i1>, %arg3: tensor<i1>, %arg4: tensor<44xf32>) -> (tensor<108xi64>, tensor<36xi64>, tensor<i1>, tensor<i1>, tensor<44xf32>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<18xi64>, tensor<18xi64>) -> tensor<18xi64>
    %1 = tosa.concat %0, %0 {axis = 0 : i32} : (tensor<18xi64>, tensor<18xi64>) -> tensor<36xi64>
    %2 = tosa.logical_and %arg2, %arg3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %t_3 = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %3 = tosa.tile %1, %t_3 : (tensor<36xi64>, !tosa.shape<1>) -> tensor<108xi64>
    %4 = tosa.bitwise_and %2, %2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %5 = tosa.logical_and %4, %2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %6 = tosa.reverse %1 {axis = 0 : i32} : (tensor<36xi64>) -> tensor<36xi64>
    %7 = tosa.logical_left_shift %5, %2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %8 = tosa.bitwise_not %7 : (tensor<i1>) -> tensor<i1>
    %9 = tosa.add %4, %2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %10 = tosa.tanh %arg4 : (tensor<44xf32>) -> tensor<44xf32>
    return %3, %6, %8, %9, %10 : tensor<108xi64>, tensor<36xi64>, tensor<i1>, tensor<i1>, tensor<44xf32>
  }
}
