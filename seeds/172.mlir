module {
  func.func @main(%arg0: tensor<68x19x6x70x12x28xf32>, %arg1: tensor<75x82x34xi32>) -> (tensor<68x19x6x70x12x28xi1>, tensor<68x19x6x70x12x28xf32>, tensor<68x19x6x70x12x28xf32>, tensor<225x164x102xi32>) {
    %0 = tosa.rsqrt %arg0 : (tensor<68x19x6x70x12x28xf32>) -> tensor<68x19x6x70x12x28xf32>
    %t_1 = tosa.const_shape {values = dense<[ 3, 2, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.tile %arg1, %t_1 : (tensor<75x82x34xi32>, !tosa.shape<3>) -> tensor<225x164x102xi32>
    %2 = tosa.greater_equal %0, %0 : (tensor<68x19x6x70x12x28xf32>, tensor<68x19x6x70x12x28xf32>) -> tensor<68x19x6x70x12x28xi1>
    %3 = tosa.bitwise_or %1, %1 : (tensor<225x164x102xi32>, tensor<225x164x102xi32>) -> tensor<225x164x102xi32>
    %4 = tosa.tanh %0 : (tensor<68x19x6x70x12x28xf32>) -> tensor<68x19x6x70x12x28xf32>
    %5 = tosa.rsqrt %0 : (tensor<68x19x6x70x12x28xf32>) -> tensor<68x19x6x70x12x28xf32>
    %6 = tosa.bitwise_or %3, %1 : (tensor<225x164x102xi32>, tensor<225x164x102xi32>) -> tensor<225x164x102xi32>
    return %2, %4, %5, %6 : tensor<68x19x6x70x12x28xi1>, tensor<68x19x6x70x12x28xf32>, tensor<68x19x6x70x12x28xf32>, tensor<225x164x102xi32>
  }
}
