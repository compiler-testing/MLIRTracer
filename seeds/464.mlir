module {
  func.func @main(%arg0: tensor<85xi8>, %arg1: tensor<42x65x4x97x6x83xf32>, %arg2: tensor<93x96x65x76x86x2xi1>, %arg3: tensor<1x96x65x76x86x2xi1>, %arg4: tensor<99x19xi32>, %arg5: tensor<99x19xi32>) -> (tensor<1xi8>, tensor<93x96x65x76x86x2xi1>, tensor<42x65x4x97x6x83xf32>, tensor<99x19xi1>, tensor<42x65x4x97x6x83xf32>) {
    %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<85xi8>) -> tensor<1xi8>
    %1 = tosa.add %0, %0 : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %2 = tosa.log %arg1 : (tensor<42x65x4x97x6x83xf32>) -> tensor<42x65x4x97x6x83xf32>
    %3 = tosa.logical_or %arg2, %arg3 : (tensor<93x96x65x76x86x2xi1>, tensor<1x96x65x76x86x2xi1>) -> tensor<93x96x65x76x86x2xi1>
    %4 = tosa.logical_and %3, %3 : (tensor<93x96x65x76x86x2xi1>, tensor<93x96x65x76x86x2xi1>) -> tensor<93x96x65x76x86x2xi1>
    %5 = tosa.intdiv %arg4, %arg5 : (tensor<99x19xi32>, tensor<99x19xi32>) -> tensor<99x19xi32>
    %6 = tosa.bitwise_not %4 : (tensor<93x96x65x76x86x2xi1>) -> tensor<93x96x65x76x86x2xi1>
    %7 = tosa.sigmoid %2 : (tensor<42x65x4x97x6x83xf32>) -> tensor<42x65x4x97x6x83xf32>
    %8 = tosa.greater_equal %5, %5 : (tensor<99x19xi32>, tensor<99x19xi32>) -> tensor<99x19xi1>
    %in_zp_9 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_9 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %9 = tosa.negate %2, %in_zp_9, %out_zp_9 : (tensor<42x65x4x97x6x83xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<42x65x4x97x6x83xf32>
    return %1, %6, %7, %8, %9 : tensor<1xi8>, tensor<93x96x65x76x86x2xi1>, tensor<42x65x4x97x6x83xf32>, tensor<99x19xi1>, tensor<42x65x4x97x6x83xf32>
  }
}
