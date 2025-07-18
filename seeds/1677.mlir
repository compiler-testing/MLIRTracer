module {
  func.func @main(%arg0: tensor<62x76xi64>, %arg1: tensor<1x76xi64>, %arg2: tensor<f32>) -> (tensor<1x1xi1>, tensor<f32>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<62x76xi64>, tensor<1x76xi64>) -> tensor<62x76xi1>
    %1 = tosa.sub %0, %0 : (tensor<62x76xi1>, tensor<62x76xi1>) -> tensor<62x76xi1>
    %2 = tosa.logical_xor %1, %0 : (tensor<62x76xi1>, tensor<62x76xi1>) -> tensor<62x76xi1>
    %3 = tosa.reduce_all %2 {axis = 1 : i32} : (tensor<62x76xi1>) -> tensor<62x1xi1>
    %4 = tosa.bitwise_and %3, %3 : (tensor<62x1xi1>, tensor<62x1xi1>) -> tensor<62x1xi1>
    %5 = tosa.reduce_any %4 {axis = 0 : i32} : (tensor<62x1xi1>) -> tensor<1x1xi1>
    %6 = tosa.reduce_max %5 {axis = 1 : i32} : (tensor<1x1xi1>) -> tensor<1x1xi1>
    %7 = tosa.exp %arg2 : (tensor<f32>) -> tensor<f32>
    return %6, %7 : tensor<1x1xi1>, tensor<f32>
  }
}
