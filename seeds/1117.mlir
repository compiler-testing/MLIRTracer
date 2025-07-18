module {
  func.func @main(%arg0: tensor<27xi64>, %arg1: tensor<37x77x16x99xf32>, %arg2: tensor<49x66x3x95xi1>) -> (tensor<37x77x16x99xf32>, tensor<i1>, tensor<1x66x3x95xi1>) {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<27xi64>) -> tensor<i32>
    %1 = tosa.greater_equal %0, %0 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2 = tosa.floor %arg1 : (tensor<37x77x16x99xf32>) -> tensor<37x77x16x99xf32>
    %3 = tosa.reduce_any %arg2 {axis = 0 : i32} : (tensor<49x66x3x95xi1>) -> tensor<1x66x3x95xi1>
    %4 = tosa.identity %1 : (tensor<i1>) -> tensor<i1>
    %5 = tosa.logical_right_shift %3, %3 : (tensor<1x66x3x95xi1>, tensor<1x66x3x95xi1>) -> tensor<1x66x3x95xi1>
    return %2, %4, %5 : tensor<37x77x16x99xf32>, tensor<i1>, tensor<1x66x3x95xi1>
  }
}
