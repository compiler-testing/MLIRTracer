module {
  func.func @main(%arg0: tensor<52x13xf32>, %arg1: tensor<14xi32>, %arg2: tensor<1xi32>) -> (tensor<16xi1>, tensor<14xi32>) {
    %0 = tosa.reciprocal %arg0 : (tensor<52x13xf32>) -> tensor<52x13xf32>
    %1 = tosa.greater %0, %0 : (tensor<52x13xf32>, tensor<52x13xf32>) -> tensor<52x13xi1>
    %2 = tosa.reduce_any %1 {axis = 0 : i32} : (tensor<52x13xi1>) -> tensor<1x13xi1>
    %s_3_start = tosa.const_shape {values = dense<[ 0, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_3_size = tosa.const_shape {values = dense<[ 7, 10 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.slice %2, %s_3_start, %s_3_size : (tensor<1x13xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<7x10xi1>
    %4 = tosa.intdiv %arg1, %arg2 : (tensor<14xi32>, tensor<1xi32>) -> tensor<14xi32>
    %5 = tosa.reduce_any %3 {axis = 0 : i32} : (tensor<7x10xi1>) -> tensor<1x10xi1>
    %s_6_start = tosa.const_shape {values = dense<[ 0, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_6_size = tosa.const_shape {values = dense<[ 4, 4 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %6 = tosa.slice %5, %s_6_start, %s_6_size : (tensor<1x10xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<4x4xi1>
    %r_7 = tosa.const_shape {values = dense<[ 16 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %7 = tosa.reshape %6, %r_7 : (tensor<4x4xi1>, !tosa.shape<1>) -> tensor<16xi1>
    %8 = tosa.abs %4 : (tensor<14xi32>) -> tensor<14xi32>
    return %7, %8 : tensor<16xi1>, tensor<14xi32>
  }
}
