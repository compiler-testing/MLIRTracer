module {
  func.func @main(%arg0: tensor<2x93x51x45x20x83xi16>, %arg1: tensor<6x2xi64>, %arg2: tensor<f32>, %arg3: tensor<67x51x30x15x71x17xi32>, %arg4: tensor<67x51x30x1x71x1xi32>) -> (tensor<67x51x30x15x71x17xi32>, tensor<1x1xf32>, tensor<2x93x51x45x20x83xi16>, tensor<1x1xf32>, tensor<1x2xi1>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<12xindex>} : () -> !tosa.shape<12>
    %pad_const_0 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<2x93x51x45x20x83xi16>, !tosa.shape<12>, tensor<1xi16>) -> tensor<2x93x51x45x20x83xi16>
    %1 = tosa.reciprocal %arg2 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.clamp %0 {min_val = -13 : i16, max_val = 104 : i16} : (tensor<2x93x51x45x20x83xi16>) -> tensor<2x93x51x45x20x83xi16>
    %3 = tosa.maximum %arg3, %arg4 : (tensor<67x51x30x15x71x17xi32>, tensor<67x51x30x1x71x1xi32>) -> tensor<67x51x30x15x71x17xi32>
    %r_4 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %4 = tosa.reshape %1, %r_4 : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
    %5 = tosa.reduce_product %4 {axis = 0 : i32} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %6 = tosa.reduce_sum %5 {axis = 1 : i32} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %7 = tosa.maximum %6, %6 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    %8 = tosa.minimum %7, %7 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    %in_zp_9 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_9 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %9 = tosa.negate %8, %in_zp_9, %out_zp_9 : (tensor<1x1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1xf32>
    %10 = tosa.maximum %4, %4 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    %r_11 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %11 = tosa.reshape %9, %r_11 : (tensor<1x1xf32>, !tosa.shape<2>) -> tensor<1x1xf32>
    %12 = tosa.concat %11, %5 {axis = 1 : i32} : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x2xf32>
    %13 = tosa.bitwise_not %0 : (tensor<2x93x51x45x20x83xi16>) -> tensor<2x93x51x45x20x83xi16>
    %r_14 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %14 = tosa.reshape %4, %r_14 : (tensor<1x1xf32>, !tosa.shape<2>) -> tensor<1x1xf32>
    %15 = tosa.logical_left_shift %13, %2 : (tensor<2x93x51x45x20x83xi16>, tensor<2x93x51x45x20x83xi16>) -> tensor<2x93x51x45x20x83xi16>
    %16 = tosa.reduce_max %14 {axis = 0 : i32} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %17 = tosa.greater %12, %12 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xi1>
    %18 = tosa.logical_or %17, %17 : (tensor<1x2xi1>, tensor<1x2xi1>) -> tensor<1x2xi1>
    %19 = tosa.clamp %14 {min_val = -1.100000e+01 : f32, max_val = 1.040000e+02 : f32} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %20 = tosa.clamp %19 {min_val = -1.100000e+01 : f32, max_val = 1.040000e+02 : f32} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %21 = tosa.reduce_min %19 {axis = 1 : i32} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %22 = tosa.reduce_sum %19 {axis = 0 : i32} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    return %3, %10, %15, %16, %18, %20, %21, %22 : tensor<67x51x30x15x71x17xi32>, tensor<1x1xf32>, tensor<2x93x51x45x20x83xi16>, tensor<1x1xf32>, tensor<1x2xi1>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>
  }
}
