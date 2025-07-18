module {
  func.func @main(%arg0: tensor<75x4x34x32x9x100xi32>, %arg1: tensor<1x1x1x1x1x100xi32>) -> tensor<75x4x34x32x9x100xi1> {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<75x4x34x32x9x100xi32>, tensor<1x1x1x1x1x100xi32>) -> tensor<75x4x34x32x9x100xi1>
    %1 = tosa.identity %0 : (tensor<75x4x34x32x9x100xi1>) -> tensor<75x4x34x32x9x100xi1>
    return %1 : tensor<75x4x34x32x9x100xi1>
  }
}
