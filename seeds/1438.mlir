module {
  func.func @main(%arg0: tensor<48x17xf32>, %arg1: tensor<48x17xf32>, %arg2: tensor<38x25x36x43xi1>) -> (tensor<48x17xi1>, tensor<48x17xf32>, tensor<1x1x4x1xi1>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<48x17xf32>, tensor<48x17xf32>) -> tensor<48x17xf32>
    %1 = tosa.reciprocal %0 : (tensor<48x17xf32>) -> tensor<48x17xf32>
    %2 = tosa.logical_not %arg2 : (tensor<38x25x36x43xi1>) -> tensor<38x25x36x43xi1>
    %3 = tosa.clz %2 : (tensor<38x25x36x43xi1>) -> tensor<38x25x36x43xi1>
    %4 = tosa.equal %0, %1 : (tensor<48x17xf32>, tensor<48x17xf32>) -> tensor<48x17xi1>
    %s_5_start = tosa.const_shape {values = dense<[ 26, 24, 32, 32 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_5_size = tosa.const_shape {values = dense<[ 12, 1, 4, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %5 = tosa.slice %3, %s_5_start, %s_5_size : (tensor<38x25x36x43xi1>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<12x1x4x3xi1>
    %6 = tosa.exp %0 : (tensor<48x17xf32>) -> tensor<48x17xf32>
    %7 = tosa.reduce_all %5 {axis = 3 : i32} : (tensor<12x1x4x3xi1>) -> tensor<12x1x4x1xi1>
    %8 = tosa.reduce_product %7 {axis = 0 : i32} : (tensor<12x1x4x1xi1>) -> tensor<1x1x4x1xi1>
    return %4, %6, %8 : tensor<48x17xi1>, tensor<48x17xf32>, tensor<1x1x4x1xi1>
  }
}
