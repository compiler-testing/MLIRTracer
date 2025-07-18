module {
  func.func @main(%arg0: tensor<54x65x35xf32>, %arg1: tensor<91x93x36x34x55x16xi8>, %arg2: tensor<91x1x1x34x1x16xi8>, %arg3: tensor<32x12x57x69x88xi1>, %arg4: tensor<32x12x1x69x88xi1>) -> (tensor<91x93x36x34x55x16xi8>, tensor<32x12x57x69x88xi1>, tensor<54x65x1xf32>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<54x65x35xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<54x65x35xf32>
    %1 = tosa.logical_left_shift %arg1, %arg2 : (tensor<91x93x36x34x55x16xi8>, tensor<91x1x1x34x1x16xi8>) -> tensor<91x93x36x34x55x16xi8>
    %2 = tosa.sigmoid %0 : (tensor<54x65x35xf32>) -> tensor<54x65x35xf32>
    %3 = tosa.floor %2 : (tensor<54x65x35xf32>) -> tensor<54x65x35xf32>
    %4 = tosa.logical_xor %arg3, %arg4 : (tensor<32x12x57x69x88xi1>, tensor<32x12x1x69x88xi1>) -> tensor<32x12x57x69x88xi1>
    %5 = tosa.reduce_product %3 {axis = 2 : i32} : (tensor<54x65x35xf32>) -> tensor<54x65x1xf32>
    return %1, %4, %5 : tensor<91x93x36x34x55x16xi8>, tensor<32x12x57x69x88xi1>, tensor<54x65x1xf32>
  }
}
