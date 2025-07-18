module {
  func.func @main(%arg0: tensor<100x43x75x48x91xf32>) -> tensor<100x43x75x48x91xf32> {
    %0 = tosa.sigmoid %arg0 : (tensor<100x43x75x48x91xf32>) -> tensor<100x43x75x48x91xf32>
    return %0 : tensor<100x43x75x48x91xf32>
  }
}
