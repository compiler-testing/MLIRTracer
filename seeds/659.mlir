module {
  func.func @main(%arg0: tensor<1xi8>, %arg1: tensor<2x94x2x55xf32>, %arg2: tensor<95x90x13x66xf32>, %arg3: tensor<95xf32>) -> (tensor<1xi8>, tensor<1x185x17x95xf32>, tensor<2x185x17x95xf32>) {
    %t_0 = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.tile %arg0, %t_0 : (tensor<1xi8>, !tosa.shape<1>) -> tensor<1xi8>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 1, 1, 2, 1>, stride = array<i64: 1, 1>, out_shape = array<i64: 2, 185, 17, 95>} : (tensor<2x94x2x55xf32>, tensor<95x90x13x66xf32>, tensor<95xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x185x17x95xf32>
    %2 = tosa.bitwise_and %0, %0 : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %3 = tosa.exp %1 : (tensor<2x185x17x95xf32>) -> tensor<2x185x17x95xf32>
    %4 = tosa.log %1 : (tensor<2x185x17x95xf32>) -> tensor<2x185x17x95xf32>
    %5 = tosa.reduce_max %4 {axis = 0 : i32} : (tensor<2x185x17x95xf32>) -> tensor<1x185x17x95xf32>
    %6 = tosa.minimum %3, %3 : (tensor<2x185x17x95xf32>, tensor<2x185x17x95xf32>) -> tensor<2x185x17x95xf32>
    return %2, %5, %6 : tensor<1xi8>, tensor<1x185x17x95xf32>, tensor<2x185x17x95xf32>
  }
}
