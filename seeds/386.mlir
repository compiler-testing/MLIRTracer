module {
  func.func @main(%arg0: tensor<55x38x23x49x51x55xi32>, %arg1: tensor<26x78x82xi8>, %arg2: tensor<74x57x90x5x56x18xf32>) -> (tensor<55x38x23x49x51x55xi32>, tensor<4x3x9x10x10x9xf32>, tensor<1x78x1xi8>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<55x38x23x49x51x55xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<55x38x23x49x51x55xi32>
    %1 = tosa.reduce_max %arg1 {axis = 0 : i32} : (tensor<26x78x82xi8>) -> tensor<1x78x82xi8>
    %2 = tosa.bitwise_xor %0, %0 : (tensor<55x38x23x49x51x55xi32>, tensor<55x38x23x49x51x55xi32>) -> tensor<55x38x23x49x51x55xi32>
    %3 = tosa.ceil %arg2 : (tensor<74x57x90x5x56x18xf32>) -> tensor<74x57x90x5x56x18xf32>
    %4 = tosa.ceil %3 : (tensor<74x57x90x5x56x18xf32>) -> tensor<74x57x90x5x56x18xf32>
    %s_5_start = tosa.const_shape {values = dense<[ 15, 54, 23, 0, 46, 9 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_5_size = tosa.const_shape {values = dense<[ 4, 3, 9, 10, 10, 9 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %5 = tosa.slice %4, %s_5_start, %s_5_size : (tensor<74x57x90x5x56x18xf32>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<4x3x9x10x10x9xf32>
    %6 = tosa.tanh %5 : (tensor<4x3x9x10x10x9xf32>) -> tensor<4x3x9x10x10x9xf32>
    %7 = tosa.reduce_sum %1 {axis = 2 : i32} : (tensor<1x78x82xi8>) -> tensor<1x78x1xi8>
    %8 = tosa.clz %7 : (tensor<1x78x1xi8>) -> tensor<1x78x1xi8>
    return %2, %6, %8 : tensor<55x38x23x49x51x55xi32>, tensor<4x3x9x10x10x9xf32>, tensor<1x78x1xi8>
  }
}
