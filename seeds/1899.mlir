module {
  func.func @main(%arg0: tensor<77x75x3xi1>, %arg1: tensor<77x3x89xi1>, %arg2: tensor<93x82x43x80x63xf32>, %arg3: tensor<42x71x23xi32>, %arg4: tensor<1x71x1xi32>) -> (tensor<42x71x1xi32>, tensor<42x71x23xi1>, tensor<71x23xi32>, tensor<93x82x43x80x63xf32>, tensor<1x75x89xi1>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<77x75x3xi1>, tensor<77x3x89xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<77x75x89xi1>
    %1 = tosa.floor %arg2 : (tensor<93x82x43x80x63xf32>) -> tensor<93x82x43x80x63xf32>
    %2 = tosa.logical_or %0, %0 : (tensor<77x75x89xi1>, tensor<77x75x89xi1>) -> tensor<77x75x89xi1>
    %3 = tosa.logical_not %2 : (tensor<77x75x89xi1>) -> tensor<77x75x89xi1>
    %4 = tosa.intdiv %arg3, %arg4 : (tensor<42x71x23xi32>, tensor<1x71x1xi32>) -> tensor<42x71x23xi32>
    %5 = tosa.reduce_max %4 {axis = 2 : i32} : (tensor<42x71x23xi32>) -> tensor<42x71x1xi32>
    %6 = tosa.add %4, %4 : (tensor<42x71x23xi32>, tensor<42x71x23xi32>) -> tensor<42x71x23xi32>
    %7 = tosa.equal %6, %6 : (tensor<42x71x23xi32>, tensor<42x71x23xi32>) -> tensor<42x71x23xi1>
    %8 = tosa.argmax %4 {axis = 0 : i32} : (tensor<42x71x23xi32>) -> tensor<71x23xi32>
    %9 = tosa.concat %3, %3 {axis = 0 : i32} : (tensor<77x75x89xi1>, tensor<77x75x89xi1>) -> tensor<154x75x89xi1>
    %10 = tosa.bitwise_and %8, %8 : (tensor<71x23xi32>, tensor<71x23xi32>) -> tensor<71x23xi32>
    %11 = tosa.exp %1 : (tensor<93x82x43x80x63xf32>) -> tensor<93x82x43x80x63xf32>
    %12 = tosa.reduce_all %9 {axis = 0 : i32} : (tensor<154x75x89xi1>) -> tensor<1x75x89xi1>
    return %5, %7, %10, %11, %12 : tensor<42x71x1xi32>, tensor<42x71x23xi1>, tensor<71x23xi32>, tensor<93x82x43x80x63xf32>, tensor<1x75x89xi1>
  }
}
