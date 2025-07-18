module {
  func.func @main(%arg0: tensor<86x85x72x57x62xi16>, %arg1: tensor<86x85x1x57x1xi16>, %arg2: tensor<14x34x83x89xf32>, %arg3: tensor<81x46x49x59xf32>, %arg4: tensor<81xf32>, %arg5: tensor<37x89x12xi1>) -> (tensor<86x85x72x57x62xi16>, tensor<14x82x217x81xf32>, tensor<37x12xi32>, tensor<42x1x434x81xf32>, tensor<37x89x12xi1>, tensor<1x89x12xi1>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<86x85x72x57x62xi16>, tensor<86x85x1x57x1xi16>) -> tensor<86x85x72x57x62xi16>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 1, 2, 2, 2>, stride = array<i64: 1, 2>, out_shape = array<i64: 14, 82, 217, 81>} : (tensor<14x34x83x89xf32>, tensor<81x46x49x59xf32>, tensor<81xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<14x82x217x81xf32>
    %2 = tosa.logical_right_shift %0, %0 : (tensor<86x85x72x57x62xi16>, tensor<86x85x72x57x62xi16>) -> tensor<86x85x72x57x62xi16>
    %3 = tosa.bitwise_not %2 : (tensor<86x85x72x57x62xi16>) -> tensor<86x85x72x57x62xi16>
    %4 = tosa.rsqrt %1 : (tensor<14x82x217x81xf32>) -> tensor<14x82x217x81xf32>
    %5 = tosa.pow %1, %4 : (tensor<14x82x217x81xf32>, tensor<14x82x217x81xf32>) -> tensor<14x82x217x81xf32>
    %6 = tosa.tanh %4 : (tensor<14x82x217x81xf32>) -> tensor<14x82x217x81xf32>
    %7 = tosa.maximum %5, %4 : (tensor<14x82x217x81xf32>, tensor<14x82x217x81xf32>) -> tensor<14x82x217x81xf32>
    %8 = tosa.logical_not %arg5 : (tensor<37x89x12xi1>) -> tensor<37x89x12xi1>
    %t_9 = tosa.const_shape {values = dense<[ 3, 3, 2, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %9 = tosa.tile %6, %t_9 : (tensor<14x82x217x81xf32>, !tosa.shape<4>) -> tensor<42x246x434x81xf32>
    %10 = tosa.reverse %8 {axis = 2 : i32} : (tensor<37x89x12xi1>) -> tensor<37x89x12xi1>
    %11 = tosa.rsqrt %9 : (tensor<42x246x434x81xf32>) -> tensor<42x246x434x81xf32>
    %12 = tosa.argmax %10 {axis = 1 : i32} : (tensor<37x89x12xi1>) -> tensor<37x12xi32>
    %13 = tosa.reciprocal %11 : (tensor<42x246x434x81xf32>) -> tensor<42x246x434x81xf32>
    %14 = tosa.reduce_product %13 {axis = 1 : i32} : (tensor<42x246x434x81xf32>) -> tensor<42x1x434x81xf32>
    %15 = tosa.add %10, %8 : (tensor<37x89x12xi1>, tensor<37x89x12xi1>) -> tensor<37x89x12xi1>
    %16 = tosa.clz %15 : (tensor<37x89x12xi1>) -> tensor<37x89x12xi1>
    %17 = tosa.logical_and %15, %8 : (tensor<37x89x12xi1>, tensor<37x89x12xi1>) -> tensor<37x89x12xi1>
    %18 = tosa.reduce_min %16 {axis = 0 : i32} : (tensor<37x89x12xi1>) -> tensor<1x89x12xi1>
    %19 = tosa.reverse %18 {axis = 2 : i32} : (tensor<1x89x12xi1>) -> tensor<1x89x12xi1>
    return %3, %7, %12, %14, %17, %19 : tensor<86x85x72x57x62xi16>, tensor<14x82x217x81xf32>, tensor<37x12xi32>, tensor<42x1x434x81xf32>, tensor<37x89x12xi1>, tensor<1x89x12xi1>
  }
}
