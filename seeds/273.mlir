module {
  func.func @main(%arg0: tensor<72x43x27xf32>, %arg1: tensor<38x61xi1>, %arg2: tensor<36x56x48x89x6x17xi32>, %arg3: tensor<36x1x48x1x1x1xi32>) -> (tensor<38x1xi1>, tensor<72x43x27xi1>, tensor<36x56x48x89x6x17xi32>, tensor<1x1xi1>, tensor<72x43x27xf32>, tensor<1x1xi1>, tensor<36x56x48x89x6x17xi32>, tensor<72x43x27xf32>, tensor<72x43x27xf32>, tensor<1x2x1xi32>, tensor<3x12x19x2xi1>, tensor<1x76x1xi1>) {
    %0 = tosa.clamp %arg0 {min_val = 1.100000e+01 : f32, max_val = 2.800000e+01 : f32} : (tensor<72x43x27xf32>) -> tensor<72x43x27xf32>
    %1 = tosa.floor %0 : (tensor<72x43x27xf32>) -> tensor<72x43x27xf32>
    %2 = tosa.reduce_all %arg1 {axis = 1 : i32} : (tensor<38x61xi1>) -> tensor<38x1xi1>
    %3 = tosa.maximum %1, %0 : (tensor<72x43x27xf32>, tensor<72x43x27xf32>) -> tensor<72x43x27xf32>
    %4 = tosa.reduce_max %2 {axis = 1 : i32} : (tensor<38x1xi1>) -> tensor<38x1xi1>
    %5 = tosa.logical_right_shift %4, %2 : (tensor<38x1xi1>, tensor<38x1xi1>) -> tensor<38x1xi1>
    %6 = tosa.greater_equal %3, %3 : (tensor<72x43x27xf32>, tensor<72x43x27xf32>) -> tensor<72x43x27xi1>
    %7 = tosa.intdiv %arg2, %arg3 : (tensor<36x56x48x89x6x17xi32>, tensor<36x1x48x1x1x1xi32>) -> tensor<36x56x48x89x6x17xi32>
    %8 = tosa.maximum %7, %7 : (tensor<36x56x48x89x6x17xi32>, tensor<36x56x48x89x6x17xi32>) -> tensor<36x56x48x89x6x17xi32>
    %9 = tosa.reduce_sum %4 {axis = 0 : i32} : (tensor<38x1xi1>) -> tensor<1x1xi1>
    %10 = tosa.intdiv %7, %7 : (tensor<36x56x48x89x6x17xi32>, tensor<36x56x48x89x6x17xi32>) -> tensor<36x56x48x89x6x17xi32>
    %r_11 = tosa.const_shape {values = dense<[ 1, 2, 19, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %11 = tosa.reshape %4, %r_11 : (tensor<38x1xi1>, !tosa.shape<4>) -> tensor<1x2x19x1xi1>
    %12 = tosa.reciprocal %3 : (tensor<72x43x27xf32>) -> tensor<72x43x27xf32>
    %13 = tosa.logical_not %4 : (tensor<38x1xi1>) -> tensor<38x1xi1>
    %14 = tosa.sub %10, %10 : (tensor<36x56x48x89x6x17xi32>, tensor<36x56x48x89x6x17xi32>) -> tensor<36x56x48x89x6x17xi32>
    %15 = tosa.reduce_min %13 {axis = 0 : i32} : (tensor<38x1xi1>) -> tensor<1x1xi1>
    %16 = tosa.minimum %14, %14 : (tensor<36x56x48x89x6x17xi32>, tensor<36x56x48x89x6x17xi32>) -> tensor<36x56x48x89x6x17xi32>
    %17 = tosa.maximum %16, %14 : (tensor<36x56x48x89x6x17xi32>, tensor<36x56x48x89x6x17xi32>) -> tensor<36x56x48x89x6x17xi32>
    %18 = tosa.rsqrt %0 : (tensor<72x43x27xf32>) -> tensor<72x43x27xf32>
    %19 = tosa.tanh %0 : (tensor<72x43x27xf32>) -> tensor<72x43x27xf32>
    %20 = tosa.argmax %11 {axis = 2 : i32} : (tensor<1x2x19x1xi1>) -> tensor<1x2x1xi32>
    %21 = tosa.bitwise_or %11, %11 : (tensor<1x2x19x1xi1>, tensor<1x2x19x1xi1>) -> tensor<1x2x19x1xi1>
    %22 = tosa.concat %21, %11 {axis = 1 : i32} : (tensor<1x2x19x1xi1>, tensor<1x2x19x1xi1>) -> tensor<1x4x19x1xi1>
    %23 = tosa.reverse %22 {axis = 2 : i32} : (tensor<1x4x19x1xi1>) -> tensor<1x4x19x1xi1>
    %t_24 = tosa.const_shape {values = dense<[ 3, 3, 1, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %24 = tosa.tile %23, %t_24 : (tensor<1x4x19x1xi1>, !tosa.shape<4>) -> tensor<3x12x19x2xi1>
    %r_25 = tosa.const_shape {values = dense<[ 1, 76, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %25 = tosa.reshape %23, %r_25 : (tensor<1x4x19x1xi1>, !tosa.shape<3>) -> tensor<1x76x1xi1>
    return %5, %6, %8, %9, %12, %15, %17, %18, %19, %20, %24, %25 : tensor<38x1xi1>, tensor<72x43x27xi1>, tensor<36x56x48x89x6x17xi32>, tensor<1x1xi1>, tensor<72x43x27xf32>, tensor<1x1xi1>, tensor<36x56x48x89x6x17xi32>, tensor<72x43x27xf32>, tensor<72x43x27xf32>, tensor<1x2x1xi32>, tensor<3x12x19x2xi1>, tensor<1x76x1xi1>
  }
}
