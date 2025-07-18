module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<70x72x72x75xi32>, %arg2: tensor<63x85x13xi1>, %arg3: tensor<1x85x1xi1>) -> (tensor<63x85x13xi1>, tensor<70x72x72x75xi32>, tensor<70x72x72x75xi32>, tensor<f32>) {
    %0 = tosa.exp %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.reverse %arg1 {axis = 1 : i32} : (tensor<70x72x72x75xi32>) -> tensor<70x72x72x75xi32>
    %in_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.negate %0, %in_zp_2, %out_zp_2 : (tensor<f32>, tensor<1xf32>, tensor<1xf32>) -> tensor<f32>
    %3 = tosa.logical_xor %arg2, %arg3 : (tensor<63x85x13xi1>, tensor<1x85x1xi1>) -> tensor<63x85x13xi1>
    %4 = tosa.logical_xor %3, %3 : (tensor<63x85x13xi1>, tensor<63x85x13xi1>) -> tensor<63x85x13xi1>
    %5 = tosa.logical_left_shift %1, %1 : (tensor<70x72x72x75xi32>, tensor<70x72x72x75xi32>) -> tensor<70x72x72x75xi32>
    %6 = tosa.maximum %1, %1 : (tensor<70x72x72x75xi32>, tensor<70x72x72x75xi32>) -> tensor<70x72x72x75xi32>
    %7 = tosa.reciprocal %2 : (tensor<f32>) -> tensor<f32>
    return %4, %5, %6, %7 : tensor<63x85x13xi1>, tensor<70x72x72x75xi32>, tensor<70x72x72x75xi32>, tensor<f32>
  }
}
