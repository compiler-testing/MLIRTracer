module {
  func.func @main(%arg0: tensor<38x84x78x25x15xi16>, %arg1: tensor<25xi32>, %arg2: tensor<1xi32>, %arg3: tensor<29x94x72x96xf32>, %arg4: tensor<3x15x57x77xf32>, %arg5: tensor<3xf32>, %arg6: tensor<49xi1>) -> (tensor<38x84x78x25x15xi16>, tensor<29x220x202x3xf32>, tensor<29x220x202x3xf32>, tensor<25xi32>, tensor<25xi32>, tensor<58x110x3xi32>, tensor<29x220x202x3xf32>, tensor<58x440x202x6xf32>, tensor<1xi1>, tensor<29x220x202x3xf32>, tensor<29x220x202x3xf32>, tensor<29x220x202x6xf32>) {
    %0 = tosa.clz %arg0 : (tensor<38x84x78x25x15xi16>) -> tensor<38x84x78x25x15xi16>
    %1 = tosa.intdiv %arg1, %arg2 : (tensor<25xi32>, tensor<1xi32>) -> tensor<25xi32>
    %2 = tosa.logical_right_shift %1, %1 : (tensor<25xi32>, tensor<25xi32>) -> tensor<25xi32>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg3, %arg4, %arg5, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 1, 1, 1, 2>, stride = array<i64: 1, 2>, out_shape = array<i64: 29, 110, 202, 3>} : (tensor<29x94x72x96xf32>, tensor<3x15x57x77xf32>, tensor<3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<29x110x202x3xf32>
    %4 = tosa.exp %3 : (tensor<29x110x202x3xf32>) -> tensor<29x110x202x3xf32>
    %5 = tosa.concat %3, %4 {axis = 1 : i32} : (tensor<29x110x202x3xf32>, tensor<29x110x202x3xf32>) -> tensor<29x220x202x3xf32>
    %6 = tosa.maximum %5, %5 : (tensor<29x220x202x3xf32>, tensor<29x220x202x3xf32>) -> tensor<29x220x202x3xf32>
    %7 = tosa.minimum %5, %6 : (tensor<29x220x202x3xf32>, tensor<29x220x202x3xf32>) -> tensor<29x220x202x3xf32>
    %8 = tosa.bitwise_xor %2, %2 : (tensor<25xi32>, tensor<25xi32>) -> tensor<25xi32>
    %9 = tosa.rsqrt %5 : (tensor<29x220x202x3xf32>) -> tensor<29x220x202x3xf32>
    %10 = tosa.arithmetic_right_shift %2, %2 {round = false} : (tensor<25xi32>, tensor<25xi32>) -> tensor<25xi32>
    %11 = tosa.clz %8 : (tensor<25xi32>) -> tensor<25xi32>
    %12 = tosa.reduce_any %arg6 {axis = 0 : i32} : (tensor<49xi1>) -> tensor<1xi1>
    %13 = tosa.argmax %4 {axis = 2 : i32} : (tensor<29x110x202x3xf32>) -> tensor<29x110x3xi32>
    %in_zp_14 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_14 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %14 = tosa.negate %5, %in_zp_14, %out_zp_14 : (tensor<29x220x202x3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<29x220x202x3xf32>
    %15 = tosa.reverse %13 {axis = 2 : i32} : (tensor<29x110x3xi32>) -> tensor<29x110x3xi32>
    %16 = tosa.concat %15, %15 {axis = 0 : i32} : (tensor<29x110x3xi32>, tensor<29x110x3xi32>) -> tensor<58x110x3xi32>
    %17 = tosa.intdiv %16, %16 : (tensor<58x110x3xi32>, tensor<58x110x3xi32>) -> tensor<58x110x3xi32>
    %18 = tosa.reciprocal %5 : (tensor<29x220x202x3xf32>) -> tensor<29x220x202x3xf32>
    %19 = tosa.log %18 : (tensor<29x220x202x3xf32>) -> tensor<29x220x202x3xf32>
    %t_20 = tosa.const_shape {values = dense<[ 2, 2, 1, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %20 = tosa.tile %14, %t_20 : (tensor<29x220x202x3xf32>, !tosa.shape<4>) -> tensor<58x440x202x6xf32>
    %21 = tosa.clz %12 : (tensor<1xi1>) -> tensor<1xi1>
    %22 = tosa.ceil %14 : (tensor<29x220x202x3xf32>) -> tensor<29x220x202x3xf32>
    %23 = tosa.ceil %14 : (tensor<29x220x202x3xf32>) -> tensor<29x220x202x3xf32>
    %24 = tosa.sigmoid %14 : (tensor<29x220x202x3xf32>) -> tensor<29x220x202x3xf32>
    %25 = tosa.rsqrt %22 : (tensor<29x220x202x3xf32>) -> tensor<29x220x202x3xf32>
    %26 = tosa.concat %25, %5 {axis = 3 : i32} : (tensor<29x220x202x3xf32>, tensor<29x220x202x3xf32>) -> tensor<29x220x202x6xf32>
    return %0, %7, %9, %10, %11, %17, %19, %20, %21, %23, %24, %26 : tensor<38x84x78x25x15xi16>, tensor<29x220x202x3xf32>, tensor<29x220x202x3xf32>, tensor<25xi32>, tensor<25xi32>, tensor<58x110x3xi32>, tensor<29x220x202x3xf32>, tensor<58x440x202x6xf32>, tensor<1xi1>, tensor<29x220x202x3xf32>, tensor<29x220x202x3xf32>, tensor<29x220x202x6xf32>
  }
}
