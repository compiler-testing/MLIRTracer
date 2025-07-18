module {
  func.func @main(%arg0: tensor<7x6x82x47x63xi32>, %arg1: tensor<85xi32>) -> (tensor<i32>, tensor<7x6x82x47x63xi1>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<7x6x82x47x63xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<7x6x82x47x63xi32>
    %1 = tosa.argmax %arg1 {axis = 0 : i32} : (tensor<85xi32>) -> tensor<i32>
    %2 = tosa.sub %1, %1 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %3 = tosa.greater_equal %0, %0 : (tensor<7x6x82x47x63xi32>, tensor<7x6x82x47x63xi32>) -> tensor<7x6x82x47x63xi1>
    %4 = tosa.abs %2 : (tensor<i32>) -> tensor<i32>
    %5 = tosa.bitwise_not %3 : (tensor<7x6x82x47x63xi1>) -> tensor<7x6x82x47x63xi1>
    return %4, %5 : tensor<i32>, tensor<7x6x82x47x63xi1>
  }
}
