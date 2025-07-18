module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<f32>, %arg3: tensor<31x17x16x11xi32>, %arg4: tensor<1x1x16x1xi32>) -> (tensor<31x17x16x11xi32>, tensor<f32>, tensor<i1>, tensor<31x17x16x11xi32>, tensor<i1>, tensor<f32>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %1 = tosa.floor %arg2 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.intdiv %arg3, %arg4 : (tensor<31x17x16x11xi32>, tensor<1x1x16x1xi32>) -> tensor<31x17x16x11xi32>
    %3 = tosa.logical_and %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = tosa.arithmetic_right_shift %3, %3 {round = false} : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %5 = tosa.minimum %2, %2 : (tensor<31x17x16x11xi32>, tensor<31x17x16x11xi32>) -> tensor<31x17x16x11xi32>
    %6 = tosa.arithmetic_right_shift %5, %5 {round = false} : (tensor<31x17x16x11xi32>, tensor<31x17x16x11xi32>) -> tensor<31x17x16x11xi32>
    %7 = tosa.bitwise_or %6, %6 : (tensor<31x17x16x11xi32>, tensor<31x17x16x11xi32>) -> tensor<31x17x16x11xi32>
    %8 = tosa.reciprocal %1 : (tensor<f32>) -> tensor<f32>
    %9 = tosa.reciprocal %1 : (tensor<f32>) -> tensor<f32>
    %10 = tosa.reverse %2 {axis = 2 : i32} : (tensor<31x17x16x11xi32>) -> tensor<31x17x16x11xi32>
    %11 = tosa.logical_and %0, %4 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %12 = tosa.maximum %10, %10 : (tensor<31x17x16x11xi32>, tensor<31x17x16x11xi32>) -> tensor<31x17x16x11xi32>
    %13 = tosa.floor %9 : (tensor<f32>) -> tensor<f32>
    %14 = tosa.equal %9, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %15 = tosa.ceil %13 : (tensor<f32>) -> tensor<f32>
    return %7, %8, %11, %12, %14, %15 : tensor<31x17x16x11xi32>, tensor<f32>, tensor<i1>, tensor<31x17x16x11xi32>, tensor<i1>, tensor<f32>
  }
}
