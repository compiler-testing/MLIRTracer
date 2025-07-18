module {
  func.func @main(%arg0: tensor<80x63x14xi32>, %arg1: tensor<80x63x1xi32>) -> tensor<80x63x14xi32> {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<80x63x14xi32>, tensor<80x63x1xi32>) -> tensor<80x63x14xi32>
    return %0 : tensor<80x63x14xi32>
  }
}
