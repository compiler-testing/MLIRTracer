module {
  func.func @main(%arg0: tensor<18x53x95x9x24x21xi8>, %arg1: tensor<1x1x1x1x24x21xi8>, %arg2: tensor<f32>) -> (tensor<f32>, tensor<18x53x95x9x24x21xi8>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<18x53x95x9x24x21xi8>, tensor<1x1x1x1x24x21xi8>) -> tensor<18x53x95x9x24x21xi8>
    %1 = tosa.log %arg2 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.sigmoid %1 : (tensor<f32>) -> tensor<f32>
    %3 = tosa.bitwise_and %0, %0 : (tensor<18x53x95x9x24x21xi8>, tensor<18x53x95x9x24x21xi8>) -> tensor<18x53x95x9x24x21xi8>
    return %2, %3 : tensor<f32>, tensor<18x53x95x9x24x21xi8>
  }
}
