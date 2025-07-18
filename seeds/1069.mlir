module {
  func.func @main(%arg0: tensor<87x3x80x52x63x74xf32>, %arg1: tensor<15xi32>, %arg2: tensor<1xi32>) -> (tensor<87x3x80x52x63x74xf32>, tensor<15xi32>) {
    %0 = tosa.sigmoid %arg0 : (tensor<87x3x80x52x63x74xf32>) -> tensor<87x3x80x52x63x74xf32>
    %1 = tosa.abs %0 : (tensor<87x3x80x52x63x74xf32>) -> tensor<87x3x80x52x63x74xf32>
    %2 = tosa.intdiv %arg1, %arg2 : (tensor<15xi32>, tensor<1xi32>) -> tensor<15xi32>
    return %1, %2 : tensor<87x3x80x52x63x74xf32>, tensor<15xi32>
  }
}
