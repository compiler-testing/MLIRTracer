module {
  func.func @main(%arg0: tensor<39x36x13x38x76xi32>) -> tensor<78x17784x38xi32> {
    %r_0 = tosa.const_shape {values = dense<[ 78, 17784, 38 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<39x36x13x38x76xi32>, !tosa.shape<3>) -> tensor<78x17784x38xi32>
    %in_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %1 = tosa.negate %0, %in_zp_1, %out_zp_1 : (tensor<78x17784x38xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<78x17784x38xi32>
    %2 = tosa.maximum %1, %1 : (tensor<78x17784x38xi32>, tensor<78x17784x38xi32>) -> tensor<78x17784x38xi32>
    return %2 : tensor<78x17784x38xi32>
  }
}
