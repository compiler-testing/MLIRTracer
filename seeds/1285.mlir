module {
  func.func @main(%arg0: tensor<38x81xi16>, %arg1: tensor<8x4x3x15xf32>, %arg2: tensor<75x22x100x79xf32>, %arg3: tensor<75xf32>) -> (tensor<i1>, tensor<8x31x75xi32>, tensor<8x62x106x75xf32>) {
    %r_0 = tosa.const_shape {values = dense<[ 3078 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<38x81xi16>, !tosa.shape<1>) -> tensor<3078xi16>
    %1 = tosa.argmax %0 {axis = 0 : i32} : (tensor<3078xi16>) -> tensor<i32>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 1, 2, 2, 2>, stride = array<i64: 2, 1>, out_shape = array<i64: 8, 31, 106, 75>} : (tensor<8x4x3x15xf32>, tensor<75x22x100x79xf32>, tensor<75xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<8x31x106x75xf32>
    %3 = tosa.greater_equal %1, %1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %4 = tosa.argmax %2 {axis = 2 : i32} : (tensor<8x31x106x75xf32>) -> tensor<8x31x75xi32>
    %5 = tosa.reciprocal %2 : (tensor<8x31x106x75xf32>) -> tensor<8x31x106x75xf32>
    %6 = tosa.maximum %5, %5 : (tensor<8x31x106x75xf32>, tensor<8x31x106x75xf32>) -> tensor<8x31x106x75xf32>
    %7 = tosa.concat %6, %2 {axis = 1 : i32} : (tensor<8x31x106x75xf32>, tensor<8x31x106x75xf32>) -> tensor<8x62x106x75xf32>
    return %3, %4, %7 : tensor<i1>, tensor<8x31x75xi32>, tensor<8x62x106x75xf32>
  }
}
