module {
  func.func @main(%arg0: tensor<77x61xi1>, %arg1: tensor<77x61xi1>, %arg2: tensor<42xi8>, %arg3: tensor<42xi8>, %arg4: tensor<76x25xf32>) -> (tensor<1x61xi1>, tensor<76x25xf32>, tensor<76x25xf32>, tensor<1xi1>, tensor<76x25xf32>, tensor<76x25xf32>, tensor<1xi8>, tensor<1xi1>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<77x61xi1>, tensor<77x61xi1>) -> tensor<77x61xi1>
    %1 = tosa.maximum %arg2, %arg3 : (tensor<42xi8>, tensor<42xi8>) -> tensor<42xi8>
    %2 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<77x61xi1>) -> tensor<1x61xi1>
    %3 = tosa.rsqrt %arg4 : (tensor<76x25xf32>) -> tensor<76x25xf32>
    %4 = tosa.reverse %1 {axis = 0 : i32} : (tensor<42xi8>) -> tensor<42xi8>
    %5 = tosa.bitwise_and %2, %2 : (tensor<1x61xi1>, tensor<1x61xi1>) -> tensor<1x61xi1>
    %6 = tosa.reduce_product %4 {axis = 0 : i32} : (tensor<42xi8>) -> tensor<1xi8>
    %7 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %8 = tosa.transpose %6 {perms = array<i32: 0>} : (tensor<1xi8>) -> tensor<1xi8>
    %9 = tosa.ceil %3 : (tensor<76x25xf32>) -> tensor<76x25xf32>
    %10 = tosa.pow %9, %3 : (tensor<76x25xf32>, tensor<76x25xf32>) -> tensor<76x25xf32>
    %11 = tosa.sigmoid %3 : (tensor<76x25xf32>) -> tensor<76x25xf32>
    %12 = tosa.greater %6, %6 : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi1>
    %13 = tosa.reciprocal %3 : (tensor<76x25xf32>) -> tensor<76x25xf32>
    %14 = tosa.logical_left_shift %8, %6 : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %15 = tosa.tanh %3 : (tensor<76x25xf32>) -> tensor<76x25xf32>
    %16 = tosa.bitwise_not %8 : (tensor<1xi8>) -> tensor<1xi8>
    %17 = tosa.equal %8, %14 : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi1>
    return %5, %10, %11, %12, %13, %15, %16, %17 : tensor<1x61xi1>, tensor<76x25xf32>, tensor<76x25xf32>, tensor<1xi1>, tensor<76x25xf32>, tensor<76x25xf32>, tensor<1xi8>, tensor<1xi1>
  }
}
