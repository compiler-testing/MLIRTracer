module {
  func.func @main(%arg0: tensor<15x80x73xi32>, %arg1: tensor<90xf32>, %arg2: tensor<24x83x22x22x38xi1>, %arg3: tensor<1x83x22x22x1xi1>) -> (tensor<90xf32>, tensor<36636864xi1>, tensor<12x5x2xi32>) {
    %0 = tosa.clamp %arg0 {min_val = 47 : i32, max_val = 149 : i32} : (tensor<15x80x73xi32>) -> tensor<15x80x73xi32>
    %s_1_start = tosa.const_shape {values = dense<[ 3, 14, 6 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_1_size = tosa.const_shape {values = dense<[ 12, 5, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<15x80x73xi32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<12x5x2xi32>
    %2 = tosa.exp %arg1 : (tensor<90xf32>) -> tensor<90xf32>
    %3 = tosa.logical_xor %arg2, %arg3 : (tensor<24x83x22x22x38xi1>, tensor<1x83x22x22x1xi1>) -> tensor<24x83x22x22x38xi1>
    %4 = tosa.logical_right_shift %1, %1 : (tensor<12x5x2xi32>, tensor<12x5x2xi32>) -> tensor<12x5x2xi32>
    %5 = tosa.logical_right_shift %3, %3 : (tensor<24x83x22x22x38xi1>, tensor<24x83x22x22x38xi1>) -> tensor<24x83x22x22x38xi1>
    %r_6 = tosa.const_shape {values = dense<[ 36636864 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %6 = tosa.reshape %5, %r_6 : (tensor<24x83x22x22x38xi1>, !tosa.shape<1>) -> tensor<36636864xi1>
    %7 = tosa.arithmetic_right_shift %4, %1 {round = true} : (tensor<12x5x2xi32>, tensor<12x5x2xi32>) -> tensor<12x5x2xi32>
    return %2, %6, %7 : tensor<90xf32>, tensor<36636864xi1>, tensor<12x5x2xi32>
  }
}
