module {
  func.func @main(%arg0: tensor<32xi8>, %arg1: tensor<14x19xi1>, %arg2: tensor<1x19xi1>, %arg3: tensor<6x67x80x16x96xf32>, %arg4: tensor<1x1x1x16x96xf32>) -> (tensor<14x19xi1>, tensor<4xi8>, tensor<6x67x80x16x96xf32>) {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<32xi8>) -> tensor<32xi8>
    %1 = tosa.reverse %0 {axis = 0 : i32} : (tensor<32xi8>) -> tensor<32xi8>
    %2 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %3 = tosa.transpose %1 {perms = array<i32: 0>} : (tensor<32xi8>) -> tensor<32xi8>
    %4 = tosa.reduce_max %3 {axis = 0 : i32} : (tensor<32xi8>) -> tensor<1xi8>
    %5 = tosa.concat %4, %4 {axis = 0 : i32} : (tensor<1xi8>, tensor<1xi8>) -> tensor<2xi8>
    %6 = tosa.logical_and %arg1, %arg2 : (tensor<14x19xi1>, tensor<1x19xi1>) -> tensor<14x19xi1>
    %7 = tosa.concat %5, %5 {axis = 0 : i32} : (tensor<2xi8>, tensor<2xi8>) -> tensor<4xi8>
    %8 = tosa.pow %arg3, %arg4 : (tensor<6x67x80x16x96xf32>, tensor<1x1x1x16x96xf32>) -> tensor<6x67x80x16x96xf32>
    return %6, %7, %8 : tensor<14x19xi1>, tensor<4xi8>, tensor<6x67x80x16x96xf32>
  }
}
