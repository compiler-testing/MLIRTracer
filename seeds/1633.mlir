module {
  func.func @main(%arg0: tensor<2x25x21xi8>, %arg1: tensor<48x83x12x22xf32>, %arg2: tensor<9x93x46x2xf32>, %arg3: tensor<9xf32>) -> (tensor<2x75x63xi8>, tensor<48x259x59xi32>, tensor<48x259x59x1xf32>) {
    %t_0 = tosa.const_shape {values = dense<[ 1, 3, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.tile %arg0, %t_0 : (tensor<2x25x21xi8>, !tosa.shape<3>) -> tensor<2x75x63xi8>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 1>, out_shape = array<i64: 48, 259, 59, 9>} : (tensor<48x83x12x22xf32>, tensor<9x93x46x2xf32>, tensor<9xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<48x259x59x9xf32>
    %2 = tosa.rsqrt %1 : (tensor<48x259x59x9xf32>) -> tensor<48x259x59x9xf32>
    %3 = tosa.minimum %2, %2 : (tensor<48x259x59x9xf32>, tensor<48x259x59x9xf32>) -> tensor<48x259x59x9xf32>
    %4 = tosa.argmax %1 {axis = 3 : i32} : (tensor<48x259x59x9xf32>) -> tensor<48x259x59xi32>
    %5 = tosa.reduce_sum %3 {axis = 3 : i32} : (tensor<48x259x59x9xf32>) -> tensor<48x259x59x1xf32>
    %6 = tosa.tanh %5 : (tensor<48x259x59x1xf32>) -> tensor<48x259x59x1xf32>
    %7 = tosa.reciprocal %6 : (tensor<48x259x59x1xf32>) -> tensor<48x259x59x1xf32>
    return %0, %4, %7 : tensor<2x75x63xi8>, tensor<48x259x59xi32>, tensor<48x259x59x1xf32>
  }
}
