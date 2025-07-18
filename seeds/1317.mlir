module {
  func.func @main(%arg0: tensor<90x86x86x74x24x72xi32>, %arg1: tensor<1x1x1x1x1x1xi32>) -> tensor<90x86x86x74x24x72xi32> {
    %0 = tosa.add %arg0, %arg1 : (tensor<90x86x86x74x24x72xi32>, tensor<1x1x1x1x1x1xi32>) -> tensor<90x86x86x74x24x72xi32>
    return %0 : tensor<90x86x86x74x24x72xi32>
  }
}
