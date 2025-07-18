module {
  func.func @main(%arg0: tensor<5x97x71x75x43x55xi8>, %arg1: tensor<1x1x1x1x1x55xi8>) -> tensor<5x97x71x75x43x55xi1> {
    %0 = tosa.greater %arg0, %arg1 : (tensor<5x97x71x75x43x55xi8>, tensor<1x1x1x1x1x55xi8>) -> tensor<5x97x71x75x43x55xi1>
    return %0 : tensor<5x97x71x75x43x55xi1>
  }
}
