module {
  func.func @main(%arg0: tensor<16x99xi32>, %arg1: tensor<1x99xi32>, %arg2: tensor<69x48x40x5x28x74xf32>) -> (tensor<69x48x40x5x28x74xf32>, tensor<16x99xi32>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<16x99xi32>, tensor<1x99xi32>) -> tensor<16x99xi32>
    %1 = tosa.reciprocal %arg2 : (tensor<69x48x40x5x28x74xf32>) -> tensor<69x48x40x5x28x74xf32>
    %2 = tosa.minimum %0, %0 : (tensor<16x99xi32>, tensor<16x99xi32>) -> tensor<16x99xi32>
    return %1, %2 : tensor<69x48x40x5x28x74xf32>, tensor<16x99xi32>
  }
}
