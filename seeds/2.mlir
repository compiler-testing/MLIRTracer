module {
  func.func @main(%arg0: tensor<22x70xi16>) -> tensor<22x1xi16> {
    %0 = tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<22x70xi16>) -> tensor<22x1xi16>
    return %0 : tensor<22x1xi16>
  }
}
