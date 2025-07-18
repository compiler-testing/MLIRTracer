module {
  func.func @main(%arg0: tensor<97x25x14x85x20xf32>) -> tensor<7x8x12x9x7xf32> {
    %0 = tosa.exp %arg0 : (tensor<97x25x14x85x20xf32>) -> tensor<97x25x14x85x20xf32>
    %1 = tosa.identity %0 : (tensor<97x25x14x85x20xf32>) -> tensor<97x25x14x85x20xf32>
    %s_2_start = tosa.const_shape {values = dense<[ 56, 13, 2, 16, 13 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_2_size = tosa.const_shape {values = dense<[ 7, 8, 12, 9, 7 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %2 = tosa.slice %1, %s_2_start, %s_2_size : (tensor<97x25x14x85x20xf32>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<7x8x12x9x7xf32>
    %in_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.negate %2, %in_zp_3, %out_zp_3 : (tensor<7x8x12x9x7xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<7x8x12x9x7xf32>
    %4 = tosa.clamp %3 {min_val = 4.000000e+00 : f32, max_val = 8.000000e+00 : f32} : (tensor<7x8x12x9x7xf32>) -> tensor<7x8x12x9x7xf32>
    return %4 : tensor<7x8x12x9x7xf32>
  }
}
