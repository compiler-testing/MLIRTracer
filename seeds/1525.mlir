module {
  func.func @main(%arg0: tensor<19x10x39x100x18xi32>, %arg1: tensor<19x1x39x1x1xi32>, %arg2: tensor<70x23x37xi1>) -> (tensor<19x10x39x100x18xi32>, tensor<70x1x37xi1>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<19x10x39x100x18xi32>, tensor<19x1x39x1x1xi32>) -> tensor<19x10x39x100x18xi32>
    %1 = tosa.reduce_any %arg2 {axis = 1 : i32} : (tensor<70x23x37xi1>) -> tensor<70x1x37xi1>
    return %0, %1 : tensor<19x10x39x100x18xi32>, tensor<70x1x37xi1>
  }
}
