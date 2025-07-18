module {
  func.func @main(%arg0: tensor<37x82x70x52xf32>, %arg1: tensor<39x81x38x10xf32>, %arg2: tensor<39xf32>, %arg3: tensor<i16>, %arg4: tensor<i16>) -> (tensor<37x164x1x1xf32>, tensor<37x164x109x39xf32>, tensor<37x164x1x1xf32>, tensor<i16>, tensor<444x2x1xi1>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, out_shape = array<i64: 37, 164, 109, 39>} : (tensor<37x82x70x52xf32>, tensor<39x81x38x10xf32>, tensor<39xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<37x164x109x39xf32>
    %1 = tosa.reduce_sum %0 {axis = 3 : i32} : (tensor<37x164x109x39xf32>) -> tensor<37x164x109x1xf32>
    %2 = tosa.bitwise_and %arg3, %arg4 : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %3 = tosa.reduce_product %1 {axis = 2 : i32} : (tensor<37x164x109x1xf32>) -> tensor<37x164x1x1xf32>
    %4 = tosa.reduce_product %3 {axis = 2 : i32} : (tensor<37x164x1x1xf32>) -> tensor<37x164x1x1xf32>
    %5 = tosa.greater_equal %4, %3 : (tensor<37x164x1x1xf32>, tensor<37x164x1x1xf32>) -> tensor<37x164x1x1xi1>
    %6 = tosa.clamp %2 {min_val = 50 : i16, max_val = 69 : i16} : (tensor<i16>) -> tensor<i16>
    %7 = tosa.reduce_any %5 {axis = 3 : i32} : (tensor<37x164x1x1xi1>) -> tensor<37x164x1x1xi1>
    %8 = tosa.logical_not %7 : (tensor<37x164x1x1xi1>) -> tensor<37x164x1x1xi1>
    %t_9 = tosa.const_shape {values = dense<[ 3, 1, 2, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %9 = tosa.tile %8, %t_9 : (tensor<37x164x1x1xi1>, !tosa.shape<4>) -> tensor<111x164x2x1xi1>
    %10 = tosa.sigmoid %3 : (tensor<37x164x1x1xf32>) -> tensor<37x164x1x1xf32>
    %11 = tosa.add %6, %6 : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %12 = tosa.logical_and %9, %9 : (tensor<111x164x2x1xi1>, tensor<111x164x2x1xi1>) -> tensor<111x164x2x1xi1>
    %13 = tosa.ceil %10 : (tensor<37x164x1x1xf32>) -> tensor<37x164x1x1xf32>
    %14 = tosa.exp %0 : (tensor<37x164x109x39xf32>) -> tensor<37x164x109x39xf32>
    %15 = "tosa.const"() {values = dense<[3, 1, 2, 0]> : tensor<4xi32>} : () -> tensor<4xi32>
    %16 = tosa.transpose %12 {perms = array<i32: 3, 1, 2, 0>} : (tensor<111x164x2x1xi1>) -> tensor<1x164x2x111xi1>
    %r_17 = tosa.const_shape {values = dense<[ 444, 2, 41 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %17 = tosa.reshape %16, %r_17 : (tensor<1x164x2x111xi1>, !tosa.shape<3>) -> tensor<444x2x41xi1>
    %18 = tosa.abs %17 : (tensor<444x2x41xi1>) -> tensor<444x2x41xi1>
    %19 = tosa.maximum %10, %10 : (tensor<37x164x1x1xf32>, tensor<37x164x1x1xf32>) -> tensor<37x164x1x1xf32>
    %20 = tosa.bitwise_or %11, %6 : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %21 = tosa.logical_xor %18, %17 : (tensor<444x2x41xi1>, tensor<444x2x41xi1>) -> tensor<444x2x41xi1>
    %22 = tosa.logical_xor %21, %21 : (tensor<444x2x41xi1>, tensor<444x2x41xi1>) -> tensor<444x2x41xi1>
    %23 = tosa.reduce_sum %22 {axis = 2 : i32} : (tensor<444x2x41xi1>) -> tensor<444x2x1xi1>
    return %13, %14, %19, %20, %23 : tensor<37x164x1x1xf32>, tensor<37x164x109x39xf32>, tensor<37x164x1x1xf32>, tensor<i16>, tensor<444x2x1xi1>
  }
}
