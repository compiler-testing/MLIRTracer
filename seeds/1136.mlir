module {
  func.func @main(%arg0: tensor<35x22x9x63xf32>, %arg1: tensor<6x76x1x42xf32>, %arg2: tensor<6xf32>, %arg3: tensor<25x64x41x92xi1>, %arg4: tensor<25x64x1x1xi1>, %arg5: tensor<61x58x81x100xi32>, %arg6: tensor<61x1x81x1xi32>) -> (tensor<6x6x8x3xf32>, tensor<61x58x81x100xi32>, tensor<25x1x41x92xi1>, tensor<61x58x81x100xi32>, tensor<61x58x81x100xi32>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 2, 2>, out_shape = array<i64: 35, 122, 21, 6>} : (tensor<35x22x9x63xf32>, tensor<6x76x1x42xf32>, tensor<6xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<35x122x21x6xf32>
    %1 = tosa.logical_and %arg3, %arg4 : (tensor<25x64x41x92xi1>, tensor<25x64x1x1xi1>) -> tensor<25x64x41x92xi1>
    %2 = tosa.identity %0 : (tensor<35x122x21x6xf32>) -> tensor<35x122x21x6xf32>
    %3 = tosa.intdiv %arg5, %arg6 : (tensor<61x58x81x100xi32>, tensor<61x1x81x1xi32>) -> tensor<61x58x81x100xi32>
    %4 = tosa.identity %2 : (tensor<35x122x21x6xf32>) -> tensor<35x122x21x6xf32>
    %s_5_start = tosa.const_shape {values = dense<[ 5, 22, 10, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_5_size = tosa.const_shape {values = dense<[ 6, 6, 8, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %5 = tosa.slice %4, %s_5_start, %s_5_size : (tensor<35x122x21x6xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<6x6x8x3xf32>
    %6 = tosa.logical_right_shift %3, %3 : (tensor<61x58x81x100xi32>, tensor<61x58x81x100xi32>) -> tensor<61x58x81x100xi32>
    %7 = tosa.reduce_any %1 {axis = 1 : i32} : (tensor<25x64x41x92xi1>) -> tensor<25x1x41x92xi1>
    %8 = tosa.clamp %3 {min_val = 46 : i32, max_val = 99 : i32} : (tensor<61x58x81x100xi32>) -> tensor<61x58x81x100xi32>
    %9 = tosa.minimum %3, %3 : (tensor<61x58x81x100xi32>, tensor<61x58x81x100xi32>) -> tensor<61x58x81x100xi32>
    return %5, %6, %7, %8, %9 : tensor<6x6x8x3xf32>, tensor<61x58x81x100xi32>, tensor<25x1x41x92xi1>, tensor<61x58x81x100xi32>, tensor<61x58x81x100xi32>
  }
}
