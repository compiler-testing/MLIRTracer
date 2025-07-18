module {
  func.func @main(%arg0: tensor<12x5x100x81x70xf32>, %arg1: tensor<5x2xi32>, %arg2: tensor<83x22x1x97x10xi1>, %arg3: tensor<1x1x1x97x1xi1>, %arg4: tensor<16x59xi32>, %arg5: tensor<16x59xi32>) -> (tensor<12x5x100x81x70xf32>, tensor<16x59xi32>, tensor<83x22x1x97x10xi1>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<10xindex>} : () -> !tosa.shape<10>
    %pad_const_0 = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<12x5x100x81x70xf32>, !tosa.shape<10>, tensor<1xf32>) -> tensor<12x5x100x81x70xf32>
    %1 = tosa.logical_or %arg2, %arg3 : (tensor<83x22x1x97x10xi1>, tensor<1x1x1x97x1xi1>) -> tensor<83x22x1x97x10xi1>
    %2 = tosa.abs %0 : (tensor<12x5x100x81x70xf32>) -> tensor<12x5x100x81x70xf32>
    %3 = tosa.intdiv %arg4, %arg5 : (tensor<16x59xi32>, tensor<16x59xi32>) -> tensor<16x59xi32>
    %4 = tosa.arithmetic_right_shift %1, %1 {round = true} : (tensor<83x22x1x97x10xi1>, tensor<83x22x1x97x10xi1>) -> tensor<83x22x1x97x10xi1>
    return %2, %3, %4 : tensor<12x5x100x81x70xf32>, tensor<16x59xi32>, tensor<83x22x1x97x10xi1>
  }
}
