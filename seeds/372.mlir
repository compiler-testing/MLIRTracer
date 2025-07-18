module {
  func.func @main(%arg0: tensor<78x53x8x90xf32>) -> tensor<78x53x8x90xf32> {
    %in_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<78x53x8x90xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<78x53x8x90xf32>
    %1 = tosa.sigmoid %0 : (tensor<78x53x8x90xf32>) -> tensor<78x53x8x90xf32>
    %2 = tosa.maximum %1, %1 : (tensor<78x53x8x90xf32>, tensor<78x53x8x90xf32>) -> tensor<78x53x8x90xf32>
    return %2 : tensor<78x53x8x90xf32>
  }
}
