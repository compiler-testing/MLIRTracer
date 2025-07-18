module {
  func.func @main(%arg0: tensor<39x38xi32>, %arg1: tensor<39x38xi32>, %arg2: tensor<33x89xi1>, %arg3: tensor<33x1xi1>, %arg4: tensor<87x99xf32>) -> (tensor<33x89xi1>, tensor<87x99xf32>, tensor<1xi1>, tensor<87x99xf32>, tensor<99xi1>, tensor<3xi1>, tensor<i32>, tensor<87x99xf32>, tensor<1xi1>, tensor<1x1xi1>) {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<39x38xi32>, tensor<39x38xi32>) -> tensor<39x38xi32>
    %1 = tosa.bitwise_and %0, %0 : (tensor<39x38xi32>, tensor<39x38xi32>) -> tensor<39x38xi32>
    %2 = tosa.logical_and %arg2, %arg3 : (tensor<33x89xi1>, tensor<33x1xi1>) -> tensor<33x89xi1>
    %3 = tosa.exp %arg4 : (tensor<87x99xf32>) -> tensor<87x99xf32>
    %4 = tosa.argmax %3 {axis = 0 : i32} : (tensor<87x99xf32>) -> tensor<99xi32>
    %5 = tosa.reverse %4 {axis = 0 : i32} : (tensor<99xi32>) -> tensor<99xi32>
    %6 = tosa.sub %1, %0 : (tensor<39x38xi32>, tensor<39x38xi32>) -> tensor<39x38xi32>
    %7 = tosa.exp %3 : (tensor<87x99xf32>) -> tensor<87x99xf32>
    %8 = tosa.bitwise_or %6, %1 : (tensor<39x38xi32>, tensor<39x38xi32>) -> tensor<39x38xi32>
    %9 = tosa.equal %4, %5 : (tensor<99xi32>, tensor<99xi32>) -> tensor<99xi1>
    %10 = tosa.bitwise_xor %9, %9 : (tensor<99xi1>, tensor<99xi1>) -> tensor<99xi1>
    %11 = tosa.reduce_all %10 {axis = 0 : i32} : (tensor<99xi1>) -> tensor<1xi1>
    %12 = tosa.bitwise_or %9, %10 : (tensor<99xi1>, tensor<99xi1>) -> tensor<99xi1>
    %13 = tosa.sigmoid %3 : (tensor<87x99xf32>) -> tensor<87x99xf32>
    %14 = tosa.logical_or %12, %9 : (tensor<99xi1>, tensor<99xi1>) -> tensor<99xi1>
    %s_15_start = tosa.const_shape {values = dense<[ 43 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_15_size = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %15 = tosa.slice %9, %s_15_start, %s_15_size : (tensor<99xi1>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<3xi1>
    %16 = tosa.reduce_max %9 {axis = 0 : i32} : (tensor<99xi1>) -> tensor<1xi1>
    %17 = tosa.reduce_any %16 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %18 = tosa.logical_not %17 : (tensor<1xi1>) -> tensor<1xi1>
    %in_zp_19 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_19 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %19 = tosa.negate %8, %in_zp_19, %out_zp_19 : (tensor<39x38xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<39x38xi32>
    %20 = tosa.reduce_any %18 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %21 = tosa.greater %19, %1 : (tensor<39x38xi32>, tensor<39x38xi32>) -> tensor<39x38xi1>
    %22 = tosa.argmax %20 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<i32>
    %23 = tosa.sigmoid %3 : (tensor<87x99xf32>) -> tensor<87x99xf32>
    %24 = tosa.reduce_all %20 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %s_25_start = tosa.const_shape {values = dense<[ 36, 23 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_25_size = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %25 = tosa.slice %21, %s_25_start, %s_25_size : (tensor<39x38xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x1xi1>
    return %2, %7, %11, %13, %14, %15, %22, %23, %24, %25 : tensor<33x89xi1>, tensor<87x99xf32>, tensor<1xi1>, tensor<87x99xf32>, tensor<99xi1>, tensor<3xi1>, tensor<i32>, tensor<87x99xf32>, tensor<1xi1>, tensor<1x1xi1>
  }
}
