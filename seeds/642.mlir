module {
  func.func @main(%arg0: tensor<36x79x46xf32>, %arg1: tensor<3x2xi64>, %arg2: tensor<16xi1>) -> (tensor<36x79x46xf32>, tensor<36x79x46xi1>, tensor<1xi1>, tensor<3xi1>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<6xindex>} : () -> !tosa.shape<6>
    %pad_const_0 = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<36x79x46xf32>, !tosa.shape<6>, tensor<1xf32>) -> tensor<36x79x46xf32>
    %1 = tosa.logical_not %arg2 : (tensor<16xi1>) -> tensor<16xi1>
    %2 = tosa.reduce_max %1 {axis = 0 : i32} : (tensor<16xi1>) -> tensor<1xi1>
    %3 = tosa.floor %0 : (tensor<36x79x46xf32>) -> tensor<36x79x46xf32>
    %4 = tosa.logical_and %2, %2 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %s_5_start = tosa.const_shape {values = dense<[ 0 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_5_size = tosa.const_shape {values = dense<[ 10 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %5 = tosa.slice %2, %s_5_start, %s_5_size : (tensor<1xi1>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<10xi1>
    %6 = tosa.floor %0 : (tensor<36x79x46xf32>) -> tensor<36x79x46xf32>
    %7 = tosa.greater %6, %0 : (tensor<36x79x46xf32>, tensor<36x79x46xf32>) -> tensor<36x79x46xi1>
    %8 = tosa.abs %4 : (tensor<1xi1>) -> tensor<1xi1>
    %9 = tosa.add %5, %5 : (tensor<10xi1>, tensor<10xi1>) -> tensor<10xi1>
    %10 = tosa.reduce_sum %9 {axis = 0 : i32} : (tensor<10xi1>) -> tensor<1xi1>
    %t_11 = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %11 = tosa.tile %8, %t_11 : (tensor<1xi1>, !tosa.shape<1>) -> tensor<3xi1>
    %12 = tosa.clz %11 : (tensor<3xi1>) -> tensor<3xi1>
    return %3, %7, %10, %12 : tensor<36x79x46xf32>, tensor<36x79x46xi1>, tensor<1xi1>, tensor<3xi1>
  }
}
