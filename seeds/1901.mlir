module {
  func.func @main(%arg0: tensor<68x27xi64>, %arg1: tensor<13x22x2x80x77x10xf32>) -> (tensor<13x22x2x80x77x10xf32>, tensor<1xi1>) {
    %s_0_start = tosa.const_shape {values = dense<[ 61, 16 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_0_size = tosa.const_shape {values = dense<[ 7, 11 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<68x27xi64>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<7x11xi64>
    %1 = tosa.argmax %0 {axis = 1 : i32} : (tensor<7x11xi64>) -> tensor<7xi32>
    %2 = tosa.reduce_min %1 {axis = 0 : i32} : (tensor<7xi32>) -> tensor<1xi32>
    %3 = tosa.equal %2, %2 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %4 = tosa.reciprocal %arg1 : (tensor<13x22x2x80x77x10xf32>) -> tensor<13x22x2x80x77x10xf32>
    %5 = tosa.reduce_product %3 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %6 = tosa.add %5, %5 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    return %4, %6 : tensor<13x22x2x80x77x10xf32>, tensor<1xi1>
  }
}
