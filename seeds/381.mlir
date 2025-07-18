module {
  func.func @main(%arg0: tensor<56x57x65x94x18xf32>, %arg1: tensor<i64>, %arg2: tensor<83x29xi1>) -> (tensor<56x57x65x94x18xf32>, tensor<i64>, tensor<1xi32>) {
    %0 = tosa.exp %arg0 : (tensor<56x57x65x94x18xf32>) -> tensor<56x57x65x94x18xf32>
    %1 = tosa.identity %0 : (tensor<56x57x65x94x18xf32>) -> tensor<56x57x65x94x18xf32>
    %2 = tosa.bitwise_not %arg1 : (tensor<i64>) -> tensor<i64>
    %in_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %out_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %3 = tosa.negate %2, %in_zp_3, %out_zp_3 : (tensor<i64>, tensor<1xi64>, tensor<1xi64>) -> tensor<i64>
    %4 = tosa.argmax %arg2 {axis = 1 : i32} : (tensor<83x29xi1>) -> tensor<83xi32>
    %5 = tosa.reduce_product %4 {axis = 0 : i32} : (tensor<83xi32>) -> tensor<1xi32>
    return %1, %3, %5 : tensor<56x57x65x94x18xf32>, tensor<i64>, tensor<1xi32>
  }
}
