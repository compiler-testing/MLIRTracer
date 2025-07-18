module {
  func.func @main(%arg0: tensor<61xi1>, %arg1: tensor<76x82x82x35xf32>) -> (tensor<1xi1>, tensor<76x82x82x35xf32>) {
    %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<61xi1>) -> tensor<1xi1>
    %1 = tosa.logical_left_shift %0, %0 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %2 = tosa.clz %1 : (tensor<1xi1>) -> tensor<1xi1>
    %3 = tosa.reciprocal %arg1 : (tensor<76x82x82x35xf32>) -> tensor<76x82x82x35xf32>
    %4 = tosa.reverse %2 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %5 = tosa.sub %3, %3 : (tensor<76x82x82x35xf32>, tensor<76x82x82x35xf32>) -> tensor<76x82x82x35xf32>
    return %4, %5 : tensor<1xi1>, tensor<76x82x82x35xf32>
  }
}
