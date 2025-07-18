module {
  func.func @main(%arg0: tensor<64x26x97x56x78xi1>) -> tensor<64x26x97x56x78xi1> {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<64x26x97x56x78xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<64x26x97x56x78xi1>
    return %0 : tensor<64x26x97x56x78xi1>
  }
}
