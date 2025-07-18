module {
  func.func @main(%arg0: tensor<7x85xi64>, %arg1: tensor<1x1xi64>, %arg2: tensor<93xi1>, %arg3: tensor<1xi1>, %arg4: tensor<f32>) -> (tensor<7x85xi64>, tensor<f32>, tensor<7x85xi64>, tensor<7x85xi1>, tensor<1xi1>, tensor<11xi1>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<7x85xi64>, tensor<1x1xi64>) -> tensor<7x85xi64>
    %1 = tosa.maximum %0, %0 : (tensor<7x85xi64>, tensor<7x85xi64>) -> tensor<7x85xi64>
    %2 = tosa.logical_and %arg2, %arg3 : (tensor<93xi1>, tensor<1xi1>) -> tensor<93xi1>
    %3 = tosa.clz %2 : (tensor<93xi1>) -> tensor<93xi1>
    %4 = tosa.sigmoid %arg4 : (tensor<f32>) -> tensor<f32>
    %s_5_start = tosa.const_shape {values = dense<[ 63 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_5_size = tosa.const_shape {values = dense<[ 11 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %5 = tosa.slice %3, %s_5_start, %s_5_size : (tensor<93xi1>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<11xi1>
    %6 = tosa.reciprocal %4 : (tensor<f32>) -> tensor<f32>
    %7 = tosa.abs %5 : (tensor<11xi1>) -> tensor<11xi1>
    %8 = tosa.maximum %0, %0 : (tensor<7x85xi64>, tensor<7x85xi64>) -> tensor<7x85xi64>
    %9 = tosa.reduce_any %5 {axis = 0 : i32} : (tensor<11xi1>) -> tensor<1xi1>
    %10 = tosa.logical_right_shift %7, %7 : (tensor<11xi1>, tensor<11xi1>) -> tensor<11xi1>
    %11 = tosa.reduce_any %10 {axis = 0 : i32} : (tensor<11xi1>) -> tensor<1xi1>
    %12 = tosa.greater_equal %0, %0 : (tensor<7x85xi64>, tensor<7x85xi64>) -> tensor<7x85xi1>
    %13 = tosa.logical_or %11, %9 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %14 = tosa.logical_xor %7, %5 : (tensor<11xi1>, tensor<11xi1>) -> tensor<11xi1>
    return %1, %6, %8, %12, %13, %14 : tensor<7x85xi64>, tensor<f32>, tensor<7x85xi64>, tensor<7x85xi1>, tensor<1xi1>, tensor<11xi1>
  }
}
