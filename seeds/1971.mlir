module {
  func.func @main(%arg0: tensor<88x61x11x79x71xi1>, %arg1: tensor<36x57x64x96x77xi32>, %arg2: tensor<1x57x1x1x77xi32>, %arg3: tensor<80xi1>, %arg4: tensor<43x64x27x80x95xf32>) -> (tensor<36x57x64x96x77xi32>, tensor<88x61x11x79x71xi1>, tensor<i32>, tensor<1xi1>, tensor<1xi1>, tensor<36x57x64x96x154xi32>, tensor<1xi1>, tensor<36x57x64x96x77xi32>, tensor<12x1x8x4x7xf32>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<88x61x11x79x71xi1>) -> tensor<88x61x11x79x71xi1>
    %1 = tosa.intdiv %arg1, %arg2 : (tensor<36x57x64x96x77xi32>, tensor<1x57x1x1x77xi32>) -> tensor<36x57x64x96x77xi32>
    %2 = tosa.reduce_all %arg3 {axis = 0 : i32} : (tensor<80xi1>) -> tensor<1xi1>
    %3 = tosa.sub %0, %0 : (tensor<88x61x11x79x71xi1>, tensor<88x61x11x79x71xi1>) -> tensor<88x61x11x79x71xi1>
    %in_zp_4 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_4 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = tosa.negate %1, %in_zp_4, %out_zp_4 : (tensor<36x57x64x96x77xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<36x57x64x96x77xi32>
    %5 = tosa.floor %arg4 : (tensor<43x64x27x80x95xf32>) -> tensor<43x64x27x80x95xf32>
    %6 = tosa.reduce_min %2 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %7 = tosa.logical_not %3 : (tensor<88x61x11x79x71xi1>) -> tensor<88x61x11x79x71xi1>
    %8 = tosa.argmax %2 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<i32>
    %9 = tosa.reduce_max %2 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %s_10_start = tosa.const_shape {values = dense<[ 11, 14, 19, 34, 5 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_10_size = tosa.const_shape {values = dense<[ 12, 1, 8, 4, 7 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %10 = tosa.slice %5, %s_10_start, %s_10_size : (tensor<43x64x27x80x95xf32>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<12x1x8x4x7xf32>
    %11 = tosa.reduce_any %6 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %12 = tosa.clamp %1 {min_val = 27 : i32, max_val = 49 : i32} : (tensor<36x57x64x96x77xi32>) -> tensor<36x57x64x96x77xi32>
    %13 = tosa.log %10 : (tensor<12x1x8x4x7xf32>) -> tensor<12x1x8x4x7xf32>
    %14 = tosa.concat %12, %1 {axis = 4 : i32} : (tensor<36x57x64x96x77xi32>, tensor<36x57x64x96x77xi32>) -> tensor<36x57x64x96x154xi32>
    %15 = tosa.logical_not %2 : (tensor<1xi1>) -> tensor<1xi1>
    %16 = tosa.rsqrt %10 : (tensor<12x1x8x4x7xf32>) -> tensor<12x1x8x4x7xf32>
    %17 = tosa.reduce_product %15 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %18 = tosa.intdiv %12, %12 : (tensor<36x57x64x96x77xi32>, tensor<36x57x64x96x77xi32>) -> tensor<36x57x64x96x77xi32>
    %19 = tosa.maximum %16, %13 : (tensor<12x1x8x4x7xf32>, tensor<12x1x8x4x7xf32>) -> tensor<12x1x8x4x7xf32>
    return %4, %7, %8, %9, %11, %14, %17, %18, %19 : tensor<36x57x64x96x77xi32>, tensor<88x61x11x79x71xi1>, tensor<i32>, tensor<1xi1>, tensor<1xi1>, tensor<36x57x64x96x154xi32>, tensor<1xi1>, tensor<36x57x64x96x77xi32>, tensor<12x1x8x4x7xf32>
  }
}
