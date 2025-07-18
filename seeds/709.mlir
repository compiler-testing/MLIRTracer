module {
  func.func @main(%arg0: tensor<83x59x14xi1>) -> tensor<1x59x14xi1> {
    %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<83x59x14xi1>) -> tensor<1x59x14xi1>
    %1 = tosa.bitwise_or %0, %0 : (tensor<1x59x14xi1>, tensor<1x59x14xi1>) -> tensor<1x59x14xi1>
    return %1 : tensor<1x59x14xi1>
  }
}
