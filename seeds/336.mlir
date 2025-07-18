module {
  func.func @main(%arg0: tensor<53x97xi32>, %arg1: tensor<i1>, %arg2: tensor<i1>, %arg3: tensor<15x36x61x97xi1>, %arg4: tensor<83x23x45xf32>) -> (tensor<5141xi1>, tensor<45x83x23xf32>, tensor<15x36x1x97xi1>, tensor<i1>) {
    %r_0 = tosa.const_shape {values = dense<[ 5141 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<53x97xi32>, !tosa.shape<1>) -> tensor<5141xi32>
    %1 = tosa.identity %0 : (tensor<5141xi32>) -> tensor<5141xi32>
    %2 = tosa.argmax %1 {axis = 0 : i32} : (tensor<5141xi32>) -> tensor<i32>
    %3 = tosa.logical_and %arg1, %arg2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = tosa.greater_equal %2, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %5 = tosa.reduce_any %arg3 {axis = 2 : i32} : (tensor<15x36x61x97xi1>) -> tensor<15x36x1x97xi1>
    %6 = tosa.minimum %0, %0 : (tensor<5141xi32>, tensor<5141xi32>) -> tensor<5141xi32>
    %7 = tosa.greater_equal %0, %6 : (tensor<5141xi32>, tensor<5141xi32>) -> tensor<5141xi1>
    %8 = tosa.bitwise_or %5, %5 : (tensor<15x36x1x97xi1>, tensor<15x36x1x97xi1>) -> tensor<15x36x1x97xi1>
    %9 = tosa.tanh %arg4 : (tensor<83x23x45xf32>) -> tensor<83x23x45xf32>
    %10 = "tosa.const"() {values = dense<[2, 0, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
    %11 = tosa.transpose %9 {perms = array<i32: 2, 0, 1>} : (tensor<83x23x45xf32>) -> tensor<45x83x23xf32>
    %12 = tosa.clz %8 : (tensor<15x36x1x97xi1>) -> tensor<15x36x1x97xi1>
    %13 = tosa.logical_and %4, %3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    return %7, %11, %12, %13 : tensor<5141xi1>, tensor<45x83x23xf32>, tensor<15x36x1x97xi1>, tensor<i1>
  }
}
