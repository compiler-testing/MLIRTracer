module {
  func.func @main(%arg0: tensor<28x76xi1>, %arg1: tensor<1x76xi1>, %arg2: tensor<41x21xf32>, %arg3: tensor<1x21xf32>, %arg4: tensor<27x63x44xi32>, %arg5: tensor<27x1x44xi32>) -> (tensor<27x63x44xi32>, tensor<41x21xf32>, tensor<27x63x44xi32>, tensor<1x1xi1>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<28x76xi1>, tensor<1x76xi1>) -> tensor<28x76xi1>
    %1 = tosa.reduce_sum %0 {axis = 1 : i32} : (tensor<28x76xi1>) -> tensor<28x1xi1>
    %2 = tosa.reduce_min %1 {axis = 0 : i32} : (tensor<28x1xi1>) -> tensor<1x1xi1>
    %3 = tosa.logical_or %2, %2 : (tensor<1x1xi1>, tensor<1x1xi1>) -> tensor<1x1xi1>
    %4 = tosa.reduce_any %3 {axis = 1 : i32} : (tensor<1x1xi1>) -> tensor<1x1xi1>
    %5 = tosa.pow %arg2, %arg3 : (tensor<41x21xf32>, tensor<1x21xf32>) -> tensor<41x21xf32>
    %6 = tosa.reduce_sum %4 {axis = 0 : i32} : (tensor<1x1xi1>) -> tensor<1x1xi1>
    %7 = tosa.intdiv %arg4, %arg5 : (tensor<27x63x44xi32>, tensor<27x1x44xi32>) -> tensor<27x63x44xi32>
    %8 = tosa.bitwise_or %7, %7 : (tensor<27x63x44xi32>, tensor<27x63x44xi32>) -> tensor<27x63x44xi32>
    %9 = tosa.floor %5 : (tensor<41x21xf32>) -> tensor<41x21xf32>
    %10 = tosa.bitwise_not %7 : (tensor<27x63x44xi32>) -> tensor<27x63x44xi32>
    %11 = tosa.reduce_any %6 {axis = 1 : i32} : (tensor<1x1xi1>) -> tensor<1x1xi1>
    %12 = tosa.abs %11 : (tensor<1x1xi1>) -> tensor<1x1xi1>
    return %8, %9, %10, %12 : tensor<27x63x44xi32>, tensor<41x21xf32>, tensor<27x63x44xi32>, tensor<1x1xi1>
  }
}
