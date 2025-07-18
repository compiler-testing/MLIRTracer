module {
  func.func @main(%arg0: tensor<33x10x57x52xi1>) -> tensor<1x10x57x52xi1> {
    %0 = tosa.reduce_min %arg0 {axis = 0 : i32} : (tensor<33x10x57x52xi1>) -> tensor<1x10x57x52xi1>
    return %0 : tensor<1x10x57x52xi1>
  }
}
