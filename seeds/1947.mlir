module {
  func.func @main(%arg0: tensor<30x34xf32>, %arg1: tensor<30x34xf32>, %arg2: tensor<61x6x95x31xi1>, %arg3: tensor<61x6x95x1xi1>) -> (tensor<61x6x1x31xi1>, tensor<1x8x12x5xi1>, tensor<30x34xf32>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<30x34xf32>, tensor<30x34xf32>) -> tensor<30x34xf32>
    %1 = tosa.logical_left_shift %arg2, %arg3 : (tensor<61x6x95x31xi1>, tensor<61x6x95x1xi1>) -> tensor<61x6x95x31xi1>
    %2 = tosa.reduce_any %1 {axis = 2 : i32} : (tensor<61x6x95x31xi1>) -> tensor<61x6x1x31xi1>
    %s_3_start = tosa.const_shape {values = dense<[ 25, 0, 43, 26 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_3_size = tosa.const_shape {values = dense<[ 4, 8, 12, 5 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %3 = tosa.slice %1, %s_3_start, %s_3_size : (tensor<61x6x95x31xi1>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<4x8x12x5xi1>
    %4 = tosa.logical_or %3, %3 : (tensor<4x8x12x5xi1>, tensor<4x8x12x5xi1>) -> tensor<4x8x12x5xi1>
    %5 = tosa.rsqrt %0 : (tensor<30x34xf32>) -> tensor<30x34xf32>
    %6 = tosa.reduce_sum %4 {axis = 0 : i32} : (tensor<4x8x12x5xi1>) -> tensor<1x8x12x5xi1>
    %7 = tosa.floor %5 : (tensor<30x34xf32>) -> tensor<30x34xf32>
    return %2, %6, %7 : tensor<61x6x1x31xi1>, tensor<1x8x12x5xi1>, tensor<30x34xf32>
  }
}
