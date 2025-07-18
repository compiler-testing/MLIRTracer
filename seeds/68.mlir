module {
  func.func @main(%arg0: tensor<86xi1>, %arg1: tensor<34x5xf32>, %arg2: tensor<1x5xf32>) -> (tensor<86xi1>, tensor<3x20xi1>, tensor<34x5xf32>) {
    %0 = tosa.clz %arg0 : (tensor<86xi1>) -> tensor<86xi1>
    %1 = tosa.maximum %arg1, %arg2 : (tensor<34x5xf32>, tensor<1x5xf32>) -> tensor<34x5xf32>
    %2 = tosa.logical_right_shift %0, %0 : (tensor<86xi1>, tensor<86xi1>) -> tensor<86xi1>
    %3 = tosa.exp %1 : (tensor<34x5xf32>) -> tensor<34x5xf32>
    %t_4 = tosa.const_shape {values = dense<[ 3, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %4 = tosa.tile %3, %t_4 : (tensor<34x5xf32>, !tosa.shape<2>) -> tensor<102x10xf32>
    %5 = tosa.greater_equal %4, %4 : (tensor<102x10xf32>, tensor<102x10xf32>) -> tensor<102x10xi1>
    %6 = tosa.reduce_any %5 {axis = 0 : i32} : (tensor<102x10xi1>) -> tensor<1x10xi1>
    %t_7 = tosa.const_shape {values = dense<[ 3, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %7 = tosa.tile %6, %t_7 : (tensor<1x10xi1>, !tosa.shape<2>) -> tensor<3x20xi1>
    %8 = tosa.sub %1, %3 : (tensor<34x5xf32>, tensor<34x5xf32>) -> tensor<34x5xf32>
    return %2, %7, %8 : tensor<86xi1>, tensor<3x20xi1>, tensor<34x5xf32>
  }
}
