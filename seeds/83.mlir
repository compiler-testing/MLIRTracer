module {
  func.func @main(%arg0: tensor<78x74x76xi1>, %arg1: tensor<92x72x38x96xf32>, %arg2: tensor<2x82x57x86xf32>, %arg3: tensor<2xf32>) -> (tensor<1x74x76xi1>, tensor<67x46x1256xf32>, tensor<1x74x76xi1>, tensor<92x157x1x2xf32>) {
    %0 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<78x74x76xi1>) -> tensor<1x74x76xi1>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 2, 2, 1, 2>, stride = array<i64: 1, 2>, out_shape = array<i64: 92, 157, 134, 2>} : (tensor<92x72x38x96xf32>, tensor<2x82x57x86xf32>, tensor<2xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<92x157x134x2xf32>
    %2 = tosa.ceil %1 : (tensor<92x157x134x2xf32>) -> tensor<92x157x134x2xf32>
    %3 = tosa.ceil %1 : (tensor<92x157x134x2xf32>) -> tensor<92x157x134x2xf32>
    %4 = tosa.log %3 : (tensor<92x157x134x2xf32>) -> tensor<92x157x134x2xf32>
    %5 = tosa.ceil %2 : (tensor<92x157x134x2xf32>) -> tensor<92x157x134x2xf32>
    %r_6 = tosa.const_shape {values = dense<[ 3870992 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %6 = tosa.reshape %4, %r_6 : (tensor<92x157x134x2xf32>, !tosa.shape<1>) -> tensor<3870992xf32>
    %7 = tosa.logical_not %0 : (tensor<1x74x76xi1>) -> tensor<1x74x76xi1>
    %8 = tosa.rsqrt %6 : (tensor<3870992xf32>) -> tensor<3870992xf32>
    %r_9 = tosa.const_shape {values = dense<[ 67, 46, 1256 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %9 = tosa.reshape %8, %r_9 : (tensor<3870992xf32>, !tosa.shape<3>) -> tensor<67x46x1256xf32>
    %10 = tosa.log %9 : (tensor<67x46x1256xf32>) -> tensor<67x46x1256xf32>
    %11 = tosa.bitwise_not %7 : (tensor<1x74x76xi1>) -> tensor<1x74x76xi1>
    %12 = tosa.clz %11 : (tensor<1x74x76xi1>) -> tensor<1x74x76xi1>
    %13 = tosa.add %10, %9 : (tensor<67x46x1256xf32>, tensor<67x46x1256xf32>) -> tensor<67x46x1256xf32>
    %14 = tosa.bitwise_xor %0, %11 : (tensor<1x74x76xi1>, tensor<1x74x76xi1>) -> tensor<1x74x76xi1>
    %15 = tosa.reduce_product %5 {axis = 2 : i32} : (tensor<92x157x134x2xf32>) -> tensor<92x157x1x2xf32>
    return %12, %13, %14, %15 : tensor<1x74x76xi1>, tensor<67x46x1256xf32>, tensor<1x74x76xi1>, tensor<92x157x1x2xf32>
  }
}
