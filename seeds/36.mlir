module {
  func.func @main(%arg0: tensor<19xi1>, %arg1: tensor<1xi1>, %arg2: tensor<i64>, %arg3: tensor<i64>, %arg4: tensor<57x51x55xi64>, %arg5: tensor<57x1x55xi64>, %arg6: tensor<42xf32>, %arg7: tensor<70x75x28xi32>, %arg8: tensor<70x1x1xi32>) -> (tensor<i1>, tensor<1xi1>, tensor<42xf32>, tensor<42xf32>, tensor<57x51x1xi1>, tensor<42xf32>, tensor<70x75x28xi32>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<19xi1>, tensor<1xi1>) -> tensor<19xi1>
    %1 = tosa.concat %0, %0 {axis = 0 : i32} : (tensor<19xi1>, tensor<19xi1>) -> tensor<38xi1>
    %2 = tosa.greater_equal %arg2, %arg3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %s_3_start = tosa.const_shape {values = dense<[ 28 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_3_size = tosa.const_shape {values = dense<[ 6 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %3 = tosa.slice %1, %s_3_start, %s_3_size : (tensor<38xi1>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<6xi1>
    %4 = tosa.bitwise_xor %3, %3 : (tensor<6xi1>, tensor<6xi1>) -> tensor<6xi1>
    %5 = tosa.equal %arg4, %arg5 : (tensor<57x51x55xi64>, tensor<57x1x55xi64>) -> tensor<57x51x55xi1>
    %6 = tosa.reduce_max %4 {axis = 0 : i32} : (tensor<6xi1>) -> tensor<1xi1>
    %7 = tosa.sigmoid %arg6 : (tensor<42xf32>) -> tensor<42xf32>
    %8 = tosa.tanh %7 : (tensor<42xf32>) -> tensor<42xf32>
    %9 = tosa.ceil %7 : (tensor<42xf32>) -> tensor<42xf32>
    %10 = tosa.reduce_all %5 {axis = 2 : i32} : (tensor<57x51x55xi1>) -> tensor<57x51x1xi1>
    %11 = tosa.bitwise_or %10, %10 : (tensor<57x51x1xi1>, tensor<57x51x1xi1>) -> tensor<57x51x1xi1>
    %12 = tosa.reduce_any %11 {axis = 2 : i32} : (tensor<57x51x1xi1>) -> tensor<57x51x1xi1>
    %13 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %14 = tosa.transpose %7 {perms = array<i32: 0>} : (tensor<42xf32>) -> tensor<42xf32>
    %15 = tosa.log %14 : (tensor<42xf32>) -> tensor<42xf32>
    %16 = tosa.intdiv %arg7, %arg8 : (tensor<70x75x28xi32>, tensor<70x1x1xi32>) -> tensor<70x75x28xi32>
    return %2, %6, %8, %9, %12, %15, %16 : tensor<i1>, tensor<1xi1>, tensor<42xf32>, tensor<42xf32>, tensor<57x51x1xi1>, tensor<42xf32>, tensor<70x75x28xi32>
  }
}
