module {
  func.func @main(%arg0: tensor<2x14x24x5xi1>, %arg1: tensor<32x46x75x15x16xi8>, %arg2: tensor<1x1x75x15x1xi8>, %arg3: tensor<f32>) -> (tensor<1x14x24x5xi1>, tensor<32x46x75x15x16xi8>, tensor<f32>) {
    %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<2x14x24x5xi1>) -> tensor<1x14x24x5xi1>
    %in_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %1 = tosa.negate %0, %in_zp_1, %out_zp_1 : (tensor<1x14x24x5xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<1x14x24x5xi1>
    %2 = tosa.sub %1, %1 : (tensor<1x14x24x5xi1>, tensor<1x14x24x5xi1>) -> tensor<1x14x24x5xi1>
    %3 = tosa.add %2, %1 : (tensor<1x14x24x5xi1>, tensor<1x14x24x5xi1>) -> tensor<1x14x24x5xi1>
    %4 = tosa.maximum %arg1, %arg2 : (tensor<32x46x75x15x16xi8>, tensor<1x1x75x15x1xi8>) -> tensor<32x46x75x15x16xi8>
    %5 = tosa.floor %arg3 : (tensor<f32>) -> tensor<f32>
    return %3, %4, %5 : tensor<1x14x24x5xi1>, tensor<32x46x75x15x16xi8>, tensor<f32>
  }
}
