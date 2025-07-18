module {
  func.func @main(%arg0: tensor<12x64x75xi1>, %arg1: tensor<1x64x1xi1>, %arg2: tensor<28x78x84x94xf32>) -> (tensor<12x1x64xi1>, tensor<12x64x75xi1>, tensor<56x78x84x94xf32>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<12x64x75xi1>, tensor<1x64x1xi1>) -> tensor<12x64x75xi1>
    %1 = tosa.logical_xor %0, %0 : (tensor<12x64x75xi1>, tensor<12x64x75xi1>) -> tensor<12x64x75xi1>
    %2 = tosa.sigmoid %arg2 : (tensor<28x78x84x94xf32>) -> tensor<28x78x84x94xf32>
    %3 = tosa.reduce_any %1 {axis = 2 : i32} : (tensor<12x64x75xi1>) -> tensor<12x64x1xi1>
    %4 = "tosa.const"() {values = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
    %5 = tosa.transpose %3 {perms = array<i32: 0, 2, 1>} : (tensor<12x64x1xi1>) -> tensor<12x1x64xi1>
    %6 = tosa.rsqrt %2 : (tensor<28x78x84x94xf32>) -> tensor<28x78x84x94xf32>
    %7 = tosa.tanh %2 : (tensor<28x78x84x94xf32>) -> tensor<28x78x84x94xf32>
    %8 = tosa.logical_not %0 : (tensor<12x64x75xi1>) -> tensor<12x64x75xi1>
    %9 = tosa.concat %6, %7 {axis = 0 : i32} : (tensor<28x78x84x94xf32>, tensor<28x78x84x94xf32>) -> tensor<56x78x84x94xf32>
    return %5, %8, %9 : tensor<12x1x64xi1>, tensor<12x64x75xi1>, tensor<56x78x84x94xf32>
  }
}
