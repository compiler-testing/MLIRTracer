module {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<59x95x34x82xf32>, %arg3: tensor<59x1x34x82xf32>) -> (tensor<i1>, tensor<i1>, tensor<59x95x34x82xf32>, tensor<9x12x1x12xf32>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1 = tosa.pow %arg2, %arg3 : (tensor<59x95x34x82xf32>, tensor<59x1x34x82xf32>) -> tensor<59x95x34x82xf32>
    %in_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %2 = tosa.negate %0, %in_zp_2, %out_zp_2 : (tensor<i1>, tensor<1xi1>, tensor<1xi1>) -> tensor<i1>
    %3 = tosa.add %1, %1 : (tensor<59x95x34x82xf32>, tensor<59x95x34x82xf32>) -> tensor<59x95x34x82xf32>
    %4 = tosa.reduce_sum %3 {axis = 1 : i32} : (tensor<59x95x34x82xf32>) -> tensor<59x1x34x82xf32>
    %5 = tosa.logical_right_shift %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %6 = tosa.reverse %4 {axis = 0 : i32} : (tensor<59x1x34x82xf32>) -> tensor<59x1x34x82xf32>
    %s_7_start = tosa.const_shape {values = dense<[ 26, 0, 25, 17 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_7_size = tosa.const_shape {values = dense<[ 9, 12, 1, 12 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %7 = tosa.slice %6, %s_7_start, %s_7_size : (tensor<59x1x34x82xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<9x12x1x12xf32>
    %8 = tosa.maximum %3, %1 : (tensor<59x95x34x82xf32>, tensor<59x95x34x82xf32>) -> tensor<59x95x34x82xf32>
    %9 = tosa.floor %7 : (tensor<9x12x1x12xf32>) -> tensor<9x12x1x12xf32>
    %10 = tosa.abs %9 : (tensor<9x12x1x12xf32>) -> tensor<9x12x1x12xf32>
    %11 = tosa.ceil %8 : (tensor<59x95x34x82xf32>) -> tensor<59x95x34x82xf32>
    %12 = tosa.reverse %10 {axis = 2 : i32} : (tensor<9x12x1x12xf32>) -> tensor<9x12x1x12xf32>
    %13 = tosa.maximum %12, %7 : (tensor<9x12x1x12xf32>, tensor<9x12x1x12xf32>) -> tensor<9x12x1x12xf32>
    return %2, %5, %11, %13 : tensor<i1>, tensor<i1>, tensor<59x95x34x82xf32>, tensor<9x12x1x12xf32>
  }
}
