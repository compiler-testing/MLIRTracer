module {
  func.func @main(%arg0: tensor<63x85x5xi1>, %arg1: tensor<1x85x5xi1>) -> tensor<63x85x5xi1> {
    %0 = tosa.add %arg0, %arg1 : (tensor<63x85x5xi1>, tensor<1x85x5xi1>) -> tensor<63x85x5xi1>
    return %0 : tensor<63x85x5xi1>
  }
}
