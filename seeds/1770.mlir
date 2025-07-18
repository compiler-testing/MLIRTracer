module {
  func.func @main(%arg0: tensor<91x83x57xi1>, %arg1: tensor<1x83x57xi1>, %arg2: tensor<32xf32>) -> (tensor<5187xi32>, tensor<32xf32>) {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<91x83x57xi1>, tensor<1x83x57xi1>) -> tensor<91x83x57xi1>
    %1 = tosa.argmax %0 {axis = 1 : i32} : (tensor<91x83x57xi1>) -> tensor<91x57xi32>
    %r_2 = tosa.const_shape {values = dense<[ 5187 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.reshape %1, %r_2 : (tensor<91x57xi32>, !tosa.shape<1>) -> tensor<5187xi32>
    %3 = tosa.reciprocal %arg2 : (tensor<32xf32>) -> tensor<32xf32>
    return %2, %3 : tensor<5187xi32>, tensor<32xf32>
  }
}
