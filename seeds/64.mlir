module {
  func.func @main(%arg0: tensor<62xi16>, %arg1: tensor<1xi16>, %arg2: tensor<12x83x15x45xi64>, %arg3: tensor<12x1x15x45xi64>, %arg4: tensor<55x59x78xi32>, %arg5: tensor<55x59x1xi32>, %arg6: tensor<11x92x92x42x8xf32>, %arg7: tensor<11x92x1x1x8xf32>) -> (tensor<12x83x15x45xi1>, tensor<62xi16>, tensor<11x92x92x42x8xi1>, tensor<1x59x78xi1>, tensor<55x59x78xi1>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<62xi16>, tensor<1xi16>) -> tensor<62xi16>
    %1 = tosa.greater_equal %arg2, %arg3 : (tensor<12x83x15x45xi64>, tensor<12x1x15x45xi64>) -> tensor<12x83x15x45xi1>
    %2 = tosa.abs %0 : (tensor<62xi16>) -> tensor<62xi16>
    %3 = tosa.greater_equal %arg4, %arg5 : (tensor<55x59x78xi32>, tensor<55x59x1xi32>) -> tensor<55x59x78xi1>
    %4 = tosa.greater_equal %arg6, %arg7 : (tensor<11x92x92x42x8xf32>, tensor<11x92x1x1x8xf32>) -> tensor<11x92x92x42x8xi1>
    %5 = tosa.clz %3 : (tensor<55x59x78xi1>) -> tensor<55x59x78xi1>
    %6 = tosa.reduce_product %5 {axis = 0 : i32} : (tensor<55x59x78xi1>) -> tensor<1x59x78xi1>
    %7 = tosa.bitwise_or %5, %5 : (tensor<55x59x78xi1>, tensor<55x59x78xi1>) -> tensor<55x59x78xi1>
    return %1, %2, %4, %6, %7 : tensor<12x83x15x45xi1>, tensor<62xi16>, tensor<11x92x92x42x8xi1>, tensor<1x59x78xi1>, tensor<55x59x78xi1>
  }
}
