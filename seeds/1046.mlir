module {
  func.func @main(%arg0: tensor<39x87xi16>, %arg1: tensor<2x2xi32>, %arg2: tensor<69xi32>, %arg3: tensor<69xi32>, %arg4: tensor<38x21x31x38xi1>, %arg5: tensor<90x95xf32>, %arg6: tensor<1x1xf32>) -> (tensor<39x87xi16>, tensor<69xi32>, tensor<1x1x31x38xi1>, tensor<i32>, tensor<1x95xf32>, tensor<90x190xf32>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
    %pad_const_0 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<39x87xi16>, !tosa.shape<4>, tensor<1xi16>) -> tensor<39x87xi16>
    %1 = tosa.minimum %arg2, %arg3 : (tensor<69xi32>, tensor<69xi32>) -> tensor<69xi32>
    %2 = tosa.logical_not %arg4 : (tensor<38x21x31x38xi1>) -> tensor<38x21x31x38xi1>
    %3 = tosa.clz %1 : (tensor<69xi32>) -> tensor<69xi32>
    %4 = tosa.argmax %1 {axis = 0 : i32} : (tensor<69xi32>) -> tensor<i32>
    %5 = tosa.pow %arg5, %arg6 : (tensor<90x95xf32>, tensor<1x1xf32>) -> tensor<90x95xf32>
    %6 = tosa.reduce_all %2 {axis = 0 : i32} : (tensor<38x21x31x38xi1>) -> tensor<1x21x31x38xi1>
    %7 = tosa.reduce_any %6 {axis = 1 : i32} : (tensor<1x21x31x38xi1>) -> tensor<1x1x31x38xi1>
    %8 = tosa.bitwise_and %4, %4 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %9 = tosa.bitwise_or %8, %4 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %10 = tosa.minimum %5, %5 : (tensor<90x95xf32>, tensor<90x95xf32>) -> tensor<90x95xf32>
    %11 = tosa.reduce_min %5 {axis = 0 : i32} : (tensor<90x95xf32>) -> tensor<1x95xf32>
    %t_12 = tosa.const_shape {values = dense<[ 1, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %12 = tosa.tile %10, %t_12 : (tensor<90x95xf32>, !tosa.shape<2>) -> tensor<90x190xf32>
    return %0, %3, %7, %9, %11, %12 : tensor<39x87xi16>, tensor<69xi32>, tensor<1x1x31x38xi1>, tensor<i32>, tensor<1x95xf32>, tensor<90x190xf32>
  }
}
