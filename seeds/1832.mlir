module {
  func.func @main(%arg0: tensor<66x46x18xi16>) -> tensor<66x1x18xi16> {
    %0 = tosa.reduce_product %arg0 {axis = 1 : i32} : (tensor<66x46x18xi16>) -> tensor<66x1x18xi16>
    return %0 : tensor<66x1x18xi16>
  }
}
