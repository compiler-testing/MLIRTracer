module {
  func.func @main(%arg0: tensor<56x15xf32>, %arg1: tensor<71x2x22x38x69xi32>, %arg2: tensor<71x2x22x1x1xi32>) -> (tensor<56x1xf32>, tensor<12x11x5x9x7xi32>, tensor<56x1xf32>, tensor<71x2x22x38x69xi32>) {
    %0 = tosa.sigmoid %arg0 : (tensor<56x15xf32>) -> tensor<56x15xf32>
    %1 = tosa.logical_left_shift %arg1, %arg2 : (tensor<71x2x22x38x69xi32>, tensor<71x2x22x1x1xi32>) -> tensor<71x2x22x38x69xi32>
    %2 = tosa.reduce_sum %0 {axis = 1 : i32} : (tensor<56x15xf32>) -> tensor<56x1xf32>
    %3 = tosa.bitwise_and %1, %1 : (tensor<71x2x22x38x69xi32>, tensor<71x2x22x38x69xi32>) -> tensor<71x2x22x38x69xi32>
    %4 = tosa.sigmoid %2 : (tensor<56x1xf32>) -> tensor<56x1xf32>
    %s_5_start = tosa.const_shape {values = dense<[ 46, 0, 17, 29, 36 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_5_size = tosa.const_shape {values = dense<[ 12, 11, 5, 9, 7 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %5 = tosa.slice %1, %s_5_start, %s_5_size : (tensor<71x2x22x38x69xi32>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<12x11x5x9x7xi32>
    %6 = tosa.sigmoid %2 : (tensor<56x1xf32>) -> tensor<56x1xf32>
    %7 = tosa.add %3, %1 : (tensor<71x2x22x38x69xi32>, tensor<71x2x22x38x69xi32>) -> tensor<71x2x22x38x69xi32>
    return %4, %5, %6, %7 : tensor<56x1xf32>, tensor<12x11x5x9x7xi32>, tensor<56x1xf32>, tensor<71x2x22x38x69xi32>
  }
}
