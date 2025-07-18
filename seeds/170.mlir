module {
  func.func @main(%arg0: tensor<16x98x21x75x58x96xi32>, %arg1: tensor<1x98x21x1x1x1xi32>) -> tensor<16x98x21x75x58x96xi1> {
    %0 = tosa.equal %arg0, %arg1 : (tensor<16x98x21x75x58x96xi32>, tensor<1x98x21x1x1x1xi32>) -> tensor<16x98x21x75x58x96xi1>
    return %0 : tensor<16x98x21x75x58x96xi1>
  }
}
