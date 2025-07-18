module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<55x65x49xi8>, %arg2: tensor<1x65x49xi8>) -> (tensor<f32>, tensor<55x65x49xi8>) {
    %0 = tosa.ceil %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.logical_left_shift %arg1, %arg2 : (tensor<55x65x49xi8>, tensor<1x65x49xi8>) -> tensor<55x65x49xi8>
    return %0, %1 : tensor<f32>, tensor<55x65x49xi8>
  }
}
