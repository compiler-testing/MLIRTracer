module {
  func.func @main(%arg0: tensor<49x41x68xf32>, %arg1: tensor<7xi8>, %arg2: tensor<i1>) -> (tensor<i1>, tensor<1x1x68xf32>, tensor<7xi1>, tensor<7xi8>) {
    %0 = tosa.ceil %arg0 : (tensor<49x41x68xf32>) -> tensor<49x41x68xf32>
    %1 = tosa.reduce_min %0 {axis = 1 : i32} : (tensor<49x41x68xf32>) -> tensor<49x1x68xf32>
    %2 = tosa.bitwise_not %arg1 : (tensor<7xi8>) -> tensor<7xi8>
    %3 = tosa.reverse %1 {axis = 2 : i32} : (tensor<49x1x68xf32>) -> tensor<49x1x68xf32>
    %4 = tosa.logical_not %arg2 : (tensor<i1>) -> tensor<i1>
    %5 = tosa.sub %2, %2 : (tensor<7xi8>, tensor<7xi8>) -> tensor<7xi8>
    %6 = tosa.logical_or %4, %4 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %7 = tosa.reduce_min %3 {axis = 0 : i32} : (tensor<49x1x68xf32>) -> tensor<1x1x68xf32>
    %8 = tosa.bitwise_or %5, %5 : (tensor<7xi8>, tensor<7xi8>) -> tensor<7xi8>
    %9 = tosa.greater_equal %2, %5 : (tensor<7xi8>, tensor<7xi8>) -> tensor<7xi1>
    %10 = tosa.add %8, %2 : (tensor<7xi8>, tensor<7xi8>) -> tensor<7xi8>
    return %6, %7, %9, %10 : tensor<i1>, tensor<1x1x68xf32>, tensor<7xi1>, tensor<7xi8>
  }
}
