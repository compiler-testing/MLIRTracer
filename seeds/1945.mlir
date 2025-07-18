module {
  func.func @main(%arg0: tensor<65x48x68x8x97x35xi32>, %arg1: tensor<1x48x1x8x97x1xi32>, %arg2: tensor<64xf32>) -> (tensor<65x48x68x8x97x35xi32>, tensor<64xf32>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<65x48x68x8x97x35xi32>, tensor<1x48x1x8x97x1xi32>) -> tensor<65x48x68x8x97x35xi32>
    %1 = tosa.exp %arg2 : (tensor<64xf32>) -> tensor<64xf32>
    return %0, %1 : tensor<65x48x68x8x97x35xi32>, tensor<64xf32>
  }
}
