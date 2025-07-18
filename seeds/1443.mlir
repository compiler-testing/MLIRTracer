module {
  func.func @main(%arg0: tensor<25x36x76xi1>, %arg1: tensor<1x1x1xi1>) -> tensor<1x36x76xi1> {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<25x36x76xi1>, tensor<1x1x1xi1>) -> tensor<25x36x76xi1>
    %1 = tosa.reduce_product %0 {axis = 0 : i32} : (tensor<25x36x76xi1>) -> tensor<1x36x76xi1>
    return %1 : tensor<1x36x76xi1>
  }
}
