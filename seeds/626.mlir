module {
  func.func @main(%arg0: tensor<82x56x76x31xi1>) -> tensor<82x56x76x31xi1> {
    %0 = tosa.logical_not %arg0 : (tensor<82x56x76x31xi1>) -> tensor<82x56x76x31xi1>
    return %0 : tensor<82x56x76x31xi1>
  }
}
