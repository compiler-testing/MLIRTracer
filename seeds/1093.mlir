module {
  func.func @main(%arg0: tensor<10x84x34xi16>, %arg1: tensor<56x8xf32>, %arg2: tensor<83x29x50x56x77xi1>, %arg3: tensor<1x29x1x1x1xi1>) -> (tensor<56x8xf32>, tensor<10x84x34xi16>, tensor<83x29x50x56x77xi1>) {
    %0 = tosa.clamp %arg0 {min_val = 3 : i16, max_val = 14 : i16} : (tensor<10x84x34xi16>) -> tensor<10x84x34xi16>
    %1 = tosa.abs %0 : (tensor<10x84x34xi16>) -> tensor<10x84x34xi16>
    %2 = tosa.arithmetic_right_shift %1, %1 {round = true} : (tensor<10x84x34xi16>, tensor<10x84x34xi16>) -> tensor<10x84x34xi16>
    %3 = tosa.reciprocal %arg1 : (tensor<56x8xf32>) -> tensor<56x8xf32>
    %4 = tosa.bitwise_or %2, %0 : (tensor<10x84x34xi16>, tensor<10x84x34xi16>) -> tensor<10x84x34xi16>
    %5 = tosa.logical_or %arg2, %arg3 : (tensor<83x29x50x56x77xi1>, tensor<1x29x1x1x1xi1>) -> tensor<83x29x50x56x77xi1>
    return %3, %4, %5 : tensor<56x8xf32>, tensor<10x84x34xi16>, tensor<83x29x50x56x77xi1>
  }
}
