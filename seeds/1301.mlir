module {
  func.func @main(%arg0: tensor<2x92x65x14x31x11xf32>, %arg1: tensor<i1>, %arg2: tensor<i1>, %arg3: tensor<79x42x96x90xi16>, %arg4: tensor<i32>, %arg5: tensor<i32>) -> (tensor<2x65x14x92x11x31xi1>, tensor<79x1x1x90xi16>, tensor<2x92x65x14x31x11xi1>, tensor<2x92x65x14x31x11xf32>, tensor<i1>, tensor<i32>) {
    %0 = tosa.clamp %arg0 {min_val = -3.200000e+01 : f32, max_val = -1.600000e+01 : f32} : (tensor<2x92x65x14x31x11xf32>) -> tensor<2x92x65x14x31x11xf32>
    %1 = tosa.logical_or %arg1, %arg2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %2 = tosa.clz %1 : (tensor<i1>) -> tensor<i1>
    %3 = "tosa.const"() {values = dense<[0, 2, 3, 1, 5, 4]> : tensor<6xi32>} : () -> tensor<6xi32>
    %4 = tosa.transpose %0 {perms = array<i32: 0, 2, 3, 1, 5, 4>} : (tensor<2x92x65x14x31x11xf32>) -> tensor<2x65x14x92x11x31xf32>
    %5 = tosa.maximum %0, %0 : (tensor<2x92x65x14x31x11xf32>, tensor<2x92x65x14x31x11xf32>) -> tensor<2x92x65x14x31x11xf32>
    %6 = tosa.reduce_min %arg3 {axis = 2 : i32} : (tensor<79x42x96x90xi16>) -> tensor<79x42x1x90xi16>
    %7 = tosa.reduce_max %6 {axis = 1 : i32} : (tensor<79x42x1x90xi16>) -> tensor<79x1x1x90xi16>
    %8 = tosa.equal %4, %4 : (tensor<2x65x14x92x11x31xf32>, tensor<2x65x14x92x11x31xf32>) -> tensor<2x65x14x92x11x31xi1>
    %in_zp_9 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_9 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %9 = tosa.negate %1, %in_zp_9, %out_zp_9 : (tensor<i1>, tensor<1xi1>, tensor<1xi1>) -> tensor<i1>
    %10 = tosa.bitwise_or %9, %2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %11 = tosa.clamp %7 {min_val = -61 : i16, max_val = -16 : i16} : (tensor<79x1x1x90xi16>) -> tensor<79x1x1x90xi16>
    %12 = tosa.greater %5, %0 : (tensor<2x92x65x14x31x11xf32>, tensor<2x92x65x14x31x11xf32>) -> tensor<2x92x65x14x31x11xi1>
    %13 = tosa.intdiv %arg4, %arg5 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %14 = tosa.log %0 : (tensor<2x92x65x14x31x11xf32>) -> tensor<2x92x65x14x31x11xf32>
    %15 = tosa.pow %0, %14 : (tensor<2x92x65x14x31x11xf32>, tensor<2x92x65x14x31x11xf32>) -> tensor<2x92x65x14x31x11xf32>
    %16 = tosa.logical_or %10, %9 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %17 = tosa.intdiv %13, %13 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %8, %11, %12, %15, %16, %17 : tensor<2x65x14x92x11x31xi1>, tensor<79x1x1x90xi16>, tensor<2x92x65x14x31x11xi1>, tensor<2x92x65x14x31x11xf32>, tensor<i1>, tensor<i32>
  }
}
