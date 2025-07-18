module {
  func.func @main(%arg0: tensor<39x16xi1>, %arg1: tensor<82x84x39x52xf32>) -> (tensor<39x16xi1>, tensor<84x1x52xi32>, tensor<624xi1>, tensor<11x9x8xi32>, tensor<82x84x39x52xf32>, tensor<11x9x12x8xf32>) {
    %0 = tosa.logical_not %arg0 : (tensor<39x16xi1>) -> tensor<39x16xi1>
    %1 = tosa.log %arg1 : (tensor<82x84x39x52xf32>) -> tensor<82x84x39x52xf32>
    %2 = tosa.reduce_min %1 {axis = 2 : i32} : (tensor<82x84x39x52xf32>) -> tensor<82x84x1x52xf32>
    %r_3 = tosa.const_shape {values = dense<[ 624 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %3 = tosa.reshape %0, %r_3 : (tensor<39x16xi1>, !tosa.shape<1>) -> tensor<624xi1>
    %4 = tosa.tanh %2 : (tensor<82x84x1x52xf32>) -> tensor<82x84x1x52xf32>
    %s_5_start = tosa.const_shape {values = dense<[ 5, 14, 0, 44 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_5_size = tosa.const_shape {values = dense<[ 11, 9, 12, 8 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %5 = tosa.slice %4, %s_5_start, %s_5_size : (tensor<82x84x1x52xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<11x9x12x8xf32>
    %6 = tosa.add %5, %5 : (tensor<11x9x12x8xf32>, tensor<11x9x12x8xf32>) -> tensor<11x9x12x8xf32>
    %in_zp_7 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_7 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %7 = tosa.negate %2, %in_zp_7, %out_zp_7 : (tensor<82x84x1x52xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<82x84x1x52xf32>
    %8 = tosa.logical_left_shift %0, %0 : (tensor<39x16xi1>, tensor<39x16xi1>) -> tensor<39x16xi1>
    %9 = tosa.argmax %7 {axis = 0 : i32} : (tensor<82x84x1x52xf32>) -> tensor<84x1x52xi32>
    %10 = tosa.abs %6 : (tensor<11x9x12x8xf32>) -> tensor<11x9x12x8xf32>
    %11 = tosa.logical_not %3 : (tensor<624xi1>) -> tensor<624xi1>
    %12 = tosa.clamp %9 {min_val = 54 : i32, max_val = 144 : i32} : (tensor<84x1x52xi32>) -> tensor<84x1x52xi32>
    %13 = tosa.abs %12 : (tensor<84x1x52xi32>) -> tensor<84x1x52xi32>
    %14 = tosa.logical_or %11, %3 : (tensor<624xi1>, tensor<624xi1>) -> tensor<624xi1>
    %15 = tosa.argmax %10 {axis = 2 : i32} : (tensor<11x9x12x8xf32>) -> tensor<11x9x8xi32>
    %16 = tosa.floor %1 : (tensor<82x84x39x52xf32>) -> tensor<82x84x39x52xf32>
    %17 = tosa.sigmoid %10 : (tensor<11x9x12x8xf32>) -> tensor<11x9x12x8xf32>
    return %8, %13, %14, %15, %16, %17 : tensor<39x16xi1>, tensor<84x1x52xi32>, tensor<624xi1>, tensor<11x9x8xi32>, tensor<82x84x39x52xf32>, tensor<11x9x12x8xf32>
  }
}
