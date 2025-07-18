module {
  func.func @main(%arg0: tensor<79x1x76x47x96xf32>) -> tensor<6x4x7x3x2xi1> {
    %0 = tosa.ceil %arg0 : (tensor<79x1x76x47x96xf32>) -> tensor<79x1x76x47x96xf32>
    %1 = tosa.exp %0 : (tensor<79x1x76x47x96xf32>) -> tensor<79x1x76x47x96xf32>
    %s_2_start = tosa.const_shape {values = dense<[ 57, 0, 2, 44, 42 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_2_size = tosa.const_shape {values = dense<[ 6, 4, 7, 3, 2 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %2 = tosa.slice %1, %s_2_start, %s_2_size : (tensor<79x1x76x47x96xf32>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<6x4x7x3x2xf32>
    %3 = tosa.clamp %2 {min_val = -5.300000e+01 : f32, max_val = 3.400000e+01 : f32} : (tensor<6x4x7x3x2xf32>) -> tensor<6x4x7x3x2xf32>
    %4 = tosa.greater %3, %3 : (tensor<6x4x7x3x2xf32>, tensor<6x4x7x3x2xf32>) -> tensor<6x4x7x3x2xi1>
    return %4 : tensor<6x4x7x3x2xi1>
  }
}
