module {
  func.func @main(%arg0: tensor<76x19xi64>, %arg1: tensor<f32>, %arg2: tensor<72x61x45xi1>, %arg3: tensor<72x1x45xi1>) -> (tensor<72x1x45xi1>, tensor<f32>, tensor<72x1x1xi1>, tensor<1xi1>, tensor<72x1xi32>, tensor<1x2x2xi1>) {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<76x19xi64>) -> tensor<19xi32>
    %1 = tosa.log %arg1 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.logical_xor %arg2, %arg3 : (tensor<72x61x45xi1>, tensor<72x1x45xi1>) -> tensor<72x61x45xi1>
    %3 = tosa.reduce_min %2 {axis = 1 : i32} : (tensor<72x61x45xi1>) -> tensor<72x1x45xi1>
    %4 = tosa.greater_equal %0, %0 : (tensor<19xi32>, tensor<19xi32>) -> tensor<19xi1>
    %5 = tosa.logical_not %3 : (tensor<72x1x45xi1>) -> tensor<72x1x45xi1>
    %6 = tosa.ceil %1 : (tensor<f32>) -> tensor<f32>
    %7 = tosa.reduce_min %3 {axis = 2 : i32} : (tensor<72x1x45xi1>) -> tensor<72x1x1xi1>
    %8 = tosa.clz %7 : (tensor<72x1x1xi1>) -> tensor<72x1x1xi1>
    %9 = tosa.argmax %7 {axis = 2 : i32} : (tensor<72x1x1xi1>) -> tensor<72x1xi32>
    %10 = tosa.logical_not %7 : (tensor<72x1x1xi1>) -> tensor<72x1x1xi1>
    %11 = tosa.reduce_max %4 {axis = 0 : i32} : (tensor<19xi1>) -> tensor<1xi1>
    %12 = tosa.reduce_all %11 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %13 = tosa.minimum %9, %9 : (tensor<72x1xi32>, tensor<72x1xi32>) -> tensor<72x1xi32>
    %t_14 = tosa.const_shape {values = dense<[ 3, 2, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %14 = tosa.tile %10, %t_14 : (tensor<72x1x1xi1>, !tosa.shape<3>) -> tensor<216x2x2xi1>
    %15 = tosa.reduce_any %14 {axis = 0 : i32} : (tensor<216x2x2xi1>) -> tensor<1x2x2xi1>
    return %5, %6, %8, %12, %13, %15 : tensor<72x1x45xi1>, tensor<f32>, tensor<72x1x1xi1>, tensor<1xi1>, tensor<72x1xi32>, tensor<1x2x2xi1>
  }
}
