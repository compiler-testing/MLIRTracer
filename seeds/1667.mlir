module {
  func.func @main(%arg0: tensor<36x21xf32>, %arg1: tensor<36x1xf32>, %arg2: tensor<46x36x49xf32>) -> (tensor<1x21xi1>, tensor<46x36x49xf32>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<36x21xf32>, tensor<36x1xf32>) -> tensor<36x21xi1>
    %1 = tosa.bitwise_not %0 : (tensor<36x21xi1>) -> tensor<36x21xi1>
    %2 = tosa.reduce_min %1 {axis = 0 : i32} : (tensor<36x21xi1>) -> tensor<1x21xi1>
    %3 = tosa.reduce_min %2 {axis = 0 : i32} : (tensor<1x21xi1>) -> tensor<1x21xi1>
    %4 = tosa.clz %3 : (tensor<1x21xi1>) -> tensor<1x21xi1>
    %5 = tosa.floor %arg2 : (tensor<46x36x49xf32>) -> tensor<46x36x49xf32>
    return %4, %5 : tensor<1x21xi1>, tensor<46x36x49xf32>
  }
}
