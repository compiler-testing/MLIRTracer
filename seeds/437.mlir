module {
  func.func @main(%arg0: tensor<74x8xi1>, %arg1: tensor<1x8xi1>) -> tensor<74x1xi1> {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<74x8xi1>, tensor<1x8xi1>) -> tensor<74x8xi1>
    %1 = tosa.reduce_min %0 {axis = 1 : i32} : (tensor<74x8xi1>) -> tensor<74x1xi1>
    return %1 : tensor<74x1xi1>
  }
}
