module {
  func.func @main(%arg0: tensor<50x36x70x72xi8>, %arg1: tensor<16x54x68x98x75x88xi1>, %arg2: tensor<1x54x1x98x1x1xi1>) -> (tensor<50x36x70x72xi8>, tensor<16x54x68x98x75x88xi1>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<50x36x70x72xi8>) -> tensor<50x36x70x72xi8>
    %1 = tosa.logical_and %arg1, %arg2 : (tensor<16x54x68x98x75x88xi1>, tensor<1x54x1x98x1x1xi1>) -> tensor<16x54x68x98x75x88xi1>
    return %0, %1 : tensor<50x36x70x72xi8>, tensor<16x54x68x98x75x88xi1>
  }
}
