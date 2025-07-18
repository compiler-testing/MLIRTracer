module {
  func.func @main(%arg0: tensor<87xi16>) -> tensor<1xi16> {
    %0 = tosa.reduce_min %arg0 {axis = 0 : i32} : (tensor<87xi16>) -> tensor<1xi16>
    return %0 : tensor<1xi16>
  }
}
