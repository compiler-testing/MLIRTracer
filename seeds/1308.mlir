module {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<21x45x49x20xi8>) -> (tensor<i32>, tensor<1x45x49xi1>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<i32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
    %1 = tosa.sub %0, %0 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = tosa.argmax %arg1 {axis = 3 : i32} : (tensor<21x45x49x20xi8>) -> tensor<21x45x49xi32>
    %3 = tosa.reduce_min %2 {axis = 0 : i32} : (tensor<21x45x49xi32>) -> tensor<1x45x49xi32>
    %4 = tosa.equal %3, %3 : (tensor<1x45x49xi32>, tensor<1x45x49xi32>) -> tensor<1x45x49xi1>
    return %1, %4 : tensor<i32>, tensor<1x45x49xi1>
  }
}
