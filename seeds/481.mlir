module {
  func.func @main(%arg0: tensor<10x90x57xi16>, %arg1: tensor<94x68x65xf32>, %arg2: tensor<91x7xi1>, %arg3: tensor<1x1xi1>) -> (tensor<10x90x1xi16>, tensor<94x65x68xf32>, tensor<91x7xi1>) {
    %0 = tosa.reduce_product %arg0 {axis = 2 : i32} : (tensor<10x90x57xi16>) -> tensor<10x90x1xi16>
    %1 = tosa.log %arg1 : (tensor<94x68x65xf32>) -> tensor<94x68x65xf32>
    %2 = "tosa.const"() {values = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
    %3 = tosa.transpose %1 {perms = array<i32: 0, 2, 1>} : (tensor<94x68x65xf32>) -> tensor<94x65x68xf32>
    %4 = tosa.logical_xor %arg2, %arg3 : (tensor<91x7xi1>, tensor<1x1xi1>) -> tensor<91x7xi1>
    return %0, %3, %4 : tensor<10x90x1xi16>, tensor<94x65x68xf32>, tensor<91x7xi1>
  }
}
