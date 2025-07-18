module {
  func.func @main(%arg0: tensor<24xf32>) -> tensor<5x2xi1> {
    %0 = tosa.reduce_product %arg0 {axis = 0 : i32} : (tensor<24xf32>) -> tensor<1xf32>
    %in_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.negate %0, %in_zp_1, %out_zp_1 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %2 = tosa.reciprocal %1 : (tensor<1xf32>) -> tensor<1xf32>
    %3 = tosa.concat %2, %1 {axis = 0 : i32} : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %4 = tosa.pow %3, %3 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %r_5 = tosa.const_shape {values = dense<[ 2, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %5 = tosa.reshape %4, %r_5 : (tensor<2xf32>, !tosa.shape<2>) -> tensor<2x1xf32>
    %6 = tosa.clamp %5 {min_val = 2.000000e+01 : f32, max_val = 1.050000e+02 : f32} : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %s_7_start = tosa.const_shape {values = dense<[ 0, 0 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_7_size = tosa.const_shape {values = dense<[ 5, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %7 = tosa.slice %6, %s_7_start, %s_7_size : (tensor<2x1xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<5x2xf32>
    %8 = tosa.greater %7, %7 : (tensor<5x2xf32>, tensor<5x2xf32>) -> tensor<5x2xi1>
    return %8 : tensor<5x2xi1>
  }
}
