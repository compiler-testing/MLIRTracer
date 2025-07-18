module {
  func.func @main(%arg0: tensor<65x56x87x55xf32>, %arg1: tensor<11x98x61x1xf32>, %arg2: tensor<11xf32>) -> tensor<7x9x2x3xf32> {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 2, 1, 2, 2>, stride = array<i64: 1, 2>, out_shape = array<i64: 65, 156, 237, 11>} : (tensor<65x56x87x55xf32>, tensor<11x98x61x1xf32>, tensor<11xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<65x156x237x11xf32>
    %s_1_start = tosa.const_shape {values = dense<[ 41, 61, 52, 8 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_1_size = tosa.const_shape {values = dense<[ 7, 9, 2, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<65x156x237x11xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<7x9x2x3xf32>
    return %1 : tensor<7x9x2x3xf32>
  }
}
