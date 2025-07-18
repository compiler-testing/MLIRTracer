module {
  func.func @main(%arg0: tensor<21xi1>, %arg1: tensor<21xi1>, %arg2: tensor<88x28x46xi64>, %arg3: tensor<88x1x46xi64>, %arg4: tensor<38x38xi64>, %arg5: tensor<38x38xi64>, %arg6: tensor<76x27xf32>, %arg7: tensor<77x84xi32>, %arg8: tensor<77x1xi32>) -> (tensor<88x28x46xi1>, tensor<38x38xi64>, tensor<1xi1>, tensor<77x84xi32>, tensor<76x27xf32>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<21xi1>, tensor<21xi1>) -> tensor<21xi1>
    %1 = tosa.greater %arg2, %arg3 : (tensor<88x28x46xi64>, tensor<88x1x46xi64>) -> tensor<88x28x46xi1>
    %2 = tosa.minimum %arg4, %arg5 : (tensor<38x38xi64>, tensor<38x38xi64>) -> tensor<38x38xi64>
    %3 = tosa.bitwise_or %2, %2 : (tensor<38x38xi64>, tensor<38x38xi64>) -> tensor<38x38xi64>
    %4 = tosa.reduce_min %0 {axis = 0 : i32} : (tensor<21xi1>) -> tensor<1xi1>
    %5 = tosa.sigmoid %arg6 : (tensor<76x27xf32>) -> tensor<76x27xf32>
    %6 = tosa.rsqrt %5 : (tensor<76x27xf32>) -> tensor<76x27xf32>
    %7 = tosa.pow %6, %5 : (tensor<76x27xf32>, tensor<76x27xf32>) -> tensor<76x27xf32>
    %8 = tosa.tanh %5 : (tensor<76x27xf32>) -> tensor<76x27xf32>
    %9 = tosa.intdiv %arg7, %arg8 : (tensor<77x84xi32>, tensor<77x1xi32>) -> tensor<77x84xi32>
    %10 = tosa.pow %8, %7 : (tensor<76x27xf32>, tensor<76x27xf32>) -> tensor<76x27xf32>
    return %1, %3, %4, %9, %10 : tensor<88x28x46xi1>, tensor<38x38xi64>, tensor<1xi1>, tensor<77x84xi32>, tensor<76x27xf32>
  }
}
