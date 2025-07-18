module {
  func.func @main(%arg0: tensor<71x81x90x43xf32>, %arg1: tensor<20x1x5x30xi1>) -> (tensor<4x5x1x4xf32>, tensor<1x1x30xi1>, tensor<1x1x1x1xi1>) {
    %0 = tosa.ceil %arg0 : (tensor<71x81x90x43xf32>) -> tensor<71x81x90x43xf32>
    %1 = tosa.reduce_sum %0 {axis = 1 : i32} : (tensor<71x81x90x43xf32>) -> tensor<71x1x90x43xf32>
    %2 = tosa.reduce_product %1 {axis = 2 : i32} : (tensor<71x1x90x43xf32>) -> tensor<71x1x1x43xf32>
    %3 = tosa.reciprocal %2 : (tensor<71x1x1x43xf32>) -> tensor<71x1x1x43xf32>
    %4 = tosa.reduce_all %arg1 {axis = 0 : i32} : (tensor<20x1x5x30xi1>) -> tensor<1x1x5x30xi1>
    %s_5_start = tosa.const_shape {values = dense<[ 67, 0, 0, 39 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_5_size = tosa.const_shape {values = dense<[ 4, 5, 1, 4 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %5 = tosa.slice %3, %s_5_start, %s_5_size : (tensor<71x1x1x43xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<4x5x1x4xf32>
    %6 = tosa.reduce_all %4 {axis = 1 : i32} : (tensor<1x1x5x30xi1>) -> tensor<1x1x5x30xi1>
    %7 = tosa.argmax %6 {axis = 2 : i32} : (tensor<1x1x5x30xi1>) -> tensor<1x1x30xi32>
    %8 = tosa.intdiv %7, %7 : (tensor<1x1x30xi32>, tensor<1x1x30xi32>) -> tensor<1x1x30xi32>
    %9 = tosa.equal %8, %8 : (tensor<1x1x30xi32>, tensor<1x1x30xi32>) -> tensor<1x1x30xi1>
    %10 = tosa.clz %4 : (tensor<1x1x5x30xi1>) -> tensor<1x1x5x30xi1>
    %11 = tosa.reduce_any %10 {axis = 2 : i32} : (tensor<1x1x5x30xi1>) -> tensor<1x1x1x30xi1>
    %12 = tosa.reduce_min %11 {axis = 3 : i32} : (tensor<1x1x1x30xi1>) -> tensor<1x1x1x1xi1>
    return %5, %9, %12 : tensor<4x5x1x4xf32>, tensor<1x1x30xi1>, tensor<1x1x1x1xi1>
  }
}
