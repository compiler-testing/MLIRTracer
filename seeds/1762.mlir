module {
  func.func @main(%arg0: tensor<73x33x97x60xi16>, %arg1: tensor<16x82x66x26xf32>) -> (tensor<73x1x97x60xi16>, tensor<16x82x66x26xf32>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<73x33x97x60xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<73x33x97x60xi16>
    %1 = tosa.reduce_max %0 {axis = 1 : i32} : (tensor<73x33x97x60xi16>) -> tensor<73x1x97x60xi16>
    %2 = tosa.reciprocal %arg1 : (tensor<16x82x66x26xf32>) -> tensor<16x82x66x26xf32>
    return %1, %2 : tensor<73x1x97x60xi16>, tensor<16x82x66x26xf32>
  }
}
