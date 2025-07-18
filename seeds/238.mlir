module {
  func.func @main(%arg0: tensor<54x65x35xi1>, %arg1: tensor<f32>) -> (tensor<54x65x35xi1>, tensor<1x1x1xf32>) {
    %0 = tosa.clz %arg0 : (tensor<54x65x35xi1>) -> tensor<54x65x35xi1>
    %1 = tosa.log %arg1 : (tensor<f32>) -> tensor<f32>
    %r_2 = tosa.const_shape {values = dense<[ 1, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.reshape %1, %r_2 : (tensor<f32>, !tosa.shape<3>) -> tensor<1x1x1xf32>
    %3 = tosa.abs %2 : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    return %0, %3 : tensor<54x65x35xi1>, tensor<1x1x1xf32>
  }
}
