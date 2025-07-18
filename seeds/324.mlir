module {
  func.func @main(%arg0: tensor<13xi1>, %arg1: tensor<13xi1>, %arg2: tensor<12x33x96x21x77xf32>) -> (tensor<1xi1>, tensor<12x33x96x21x77xf32>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<13xi1>, tensor<13xi1>) -> tensor<13xi1>
    %1 = tosa.log %arg2 : (tensor<12x33x96x21x77xf32>) -> tensor<12x33x96x21x77xf32>
    %2 = tosa.reduce_product %0 {axis = 0 : i32} : (tensor<13xi1>) -> tensor<1xi1>
    %3 = tosa.rsqrt %1 : (tensor<12x33x96x21x77xf32>) -> tensor<12x33x96x21x77xf32>
    %t_4 = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %4 = tosa.tile %2, %t_4 : (tensor<1xi1>, !tosa.shape<1>) -> tensor<1xi1>
    %5 = tosa.maximum %1, %3 : (tensor<12x33x96x21x77xf32>, tensor<12x33x96x21x77xf32>) -> tensor<12x33x96x21x77xf32>
    return %4, %5 : tensor<1xi1>, tensor<12x33x96x21x77xf32>
  }
}
