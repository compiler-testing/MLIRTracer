module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<13xi1>, %arg3: tensor<1xi1>, %arg4: tensor<10xi32>, %arg5: tensor<10xi32>) -> (tensor<i1>, tensor<1x1x5x2xi1>, tensor<10xi32>, tensor<f32>, tensor<1xi1>, tensor<1xi32>, tensor<1xi32>, tensor<10xi32>, tensor<f32>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %1 = tosa.logical_or %arg2, %arg3 : (tensor<13xi1>, tensor<1xi1>) -> tensor<13xi1>
    %2 = tosa.logical_or %1, %1 : (tensor<13xi1>, tensor<13xi1>) -> tensor<13xi1>
    %3 = tosa.clz %2 : (tensor<13xi1>) -> tensor<13xi1>
    %4 = tosa.greater_equal %0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = tosa.bitwise_and %1, %3 : (tensor<13xi1>, tensor<13xi1>) -> tensor<13xi1>
    %6 = tosa.intdiv %arg4, %arg5 : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
    %r_7 = tosa.const_shape {values = dense<[ 1, 1, 5, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %7 = tosa.reshape %6, %r_7 : (tensor<10xi32>, !tosa.shape<4>) -> tensor<1x1x5x2xi32>
    %8 = tosa.greater %7, %7 : (tensor<1x1x5x2xi32>, tensor<1x1x5x2xi32>) -> tensor<1x1x5x2xi1>
    %9 = tosa.abs %3 : (tensor<13xi1>) -> tensor<13xi1>
    %10 = tosa.add %6, %6 : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
    %11 = tosa.sigmoid %0 : (tensor<f32>) -> tensor<f32>
    %12 = tosa.logical_and %9, %5 : (tensor<13xi1>, tensor<13xi1>) -> tensor<13xi1>
    %t_13 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %13 = tosa.tile %12, %t_13 : (tensor<13xi1>, !tosa.shape<1>) -> tensor<26xi1>
    %14 = tosa.reduce_max %13 {axis = 0 : i32} : (tensor<26xi1>) -> tensor<1xi1>
    %15 = tosa.reduce_any %14 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %16 = tosa.clz %15 : (tensor<1xi1>) -> tensor<1xi1>
    %17 = tosa.reduce_sum %6 {axis = 0 : i32} : (tensor<10xi32>) -> tensor<1xi32>
    %18 = tosa.sigmoid %0 : (tensor<f32>) -> tensor<f32>
    %19 = tosa.reduce_max %6 {axis = 0 : i32} : (tensor<10xi32>) -> tensor<1xi32>
    %20 = tosa.logical_right_shift %6, %6 : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
    %21 = tosa.log %18 : (tensor<f32>) -> tensor<f32>
    return %4, %8, %10, %11, %16, %17, %19, %20, %21 : tensor<i1>, tensor<1x1x5x2xi1>, tensor<10xi32>, tensor<f32>, tensor<1xi1>, tensor<1xi32>, tensor<1xi32>, tensor<10xi32>, tensor<f32>
  }
}
