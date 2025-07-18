module {
  func.func @main(%arg0: tensor<84x93x79xi32>, %arg1: tensor<99x27x55xf32>, %arg2: tensor<6xi1>) -> (tensor<252x279x237xi32>, tensor<99x27x55xf32>, tensor<1xi1>, tensor<6xi1>, tensor<99x27x55xf32>) {
    %t_0 = tosa.const_shape {values = dense<[ 3, 3, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.tile %arg0, %t_0 : (tensor<84x93x79xi32>, !tosa.shape<3>) -> tensor<252x279x237xi32>
    %1 = tosa.maximum %0, %0 : (tensor<252x279x237xi32>, tensor<252x279x237xi32>) -> tensor<252x279x237xi32>
    %2 = tosa.sigmoid %arg1 : (tensor<99x27x55xf32>) -> tensor<99x27x55xf32>
    %3 = tosa.logical_not %arg2 : (tensor<6xi1>) -> tensor<6xi1>
    %4 = tosa.minimum %1, %0 : (tensor<252x279x237xi32>, tensor<252x279x237xi32>) -> tensor<252x279x237xi32>
    %5 = tosa.logical_not %3 : (tensor<6xi1>) -> tensor<6xi1>
    %6 = tosa.logical_or %3, %3 : (tensor<6xi1>, tensor<6xi1>) -> tensor<6xi1>
    %7 = tosa.exp %2 : (tensor<99x27x55xf32>) -> tensor<99x27x55xf32>
    %8 = tosa.minimum %2, %2 : (tensor<99x27x55xf32>, tensor<99x27x55xf32>) -> tensor<99x27x55xf32>
    %9 = tosa.reduce_max %6 {axis = 0 : i32} : (tensor<6xi1>) -> tensor<1xi1>
    %10 = tosa.logical_right_shift %5, %5 : (tensor<6xi1>, tensor<6xi1>) -> tensor<6xi1>
    %11 = tosa.clz %10 : (tensor<6xi1>) -> tensor<6xi1>
    %12 = tosa.log %8 : (tensor<99x27x55xf32>) -> tensor<99x27x55xf32>
    return %4, %7, %9, %11, %12 : tensor<252x279x237xi32>, tensor<99x27x55xf32>, tensor<1xi1>, tensor<6xi1>, tensor<99x27x55xf32>
  }
}
