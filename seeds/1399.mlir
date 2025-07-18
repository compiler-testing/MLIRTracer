module {
  func.func @main(%arg0: tensor<49xi16>) -> tensor<49xi16> {
    %0 = tosa.clz %arg0 : (tensor<49xi16>) -> tensor<49xi16>
    return %0 : tensor<49xi16>
  }
}
