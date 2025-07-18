module {
  func.func @main(%arg0: tensor<63x23xi32>) -> tensor<1x23xi32> {
    %0 = tosa.reduce_product %arg0 {axis = 0 : i32} : (tensor<63x23xi32>) -> tensor<1x23xi32>
    %1 = tosa.minimum %0, %0 : (tensor<1x23xi32>, tensor<1x23xi32>) -> tensor<1x23xi32>
    %in_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %2 = tosa.negate %1, %in_zp_2, %out_zp_2 : (tensor<1x23xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x23xi32>
    return %2 : tensor<1x23xi32>
  }
}
