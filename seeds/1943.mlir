module {
  func.func @main(%arg0: tensor<58x50x72xi8>, %arg1: tensor<39x92x68x73x72xf32>) -> (tensor<58x50x1xi8>, tensor<39x92x68x73x72xf32>) {
    %0 = tosa.reduce_max %arg0 {axis = 2 : i32} : (tensor<58x50x72xi8>) -> tensor<58x50x1xi8>
    %in_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %1 = tosa.negate %0, %in_zp_1, %out_zp_1 : (tensor<58x50x1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<58x50x1xi8>
    %2 = tosa.rsqrt %arg1 : (tensor<39x92x68x73x72xf32>) -> tensor<39x92x68x73x72xf32>
    return %1, %2 : tensor<58x50x1xi8>, tensor<39x92x68x73x72xf32>
  }
}
