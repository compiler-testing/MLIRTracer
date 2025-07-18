module {
  func.func @main(%arg0: tensor<73x6xi1>, %arg1: tensor<91x63x29xi1>, %arg2: tensor<25x40x1x79xf32>) -> (tensor<1x1xi32>, tensor<25x40x1x79xf32>, tensor<1x1x29xi1>) {
    %0 = tosa.argmax %arg0 {axis = 1 : i32} : (tensor<73x6xi1>) -> tensor<73xi32>
    %1 = tosa.argmax %0 {axis = 0 : i32} : (tensor<73xi32>) -> tensor<i32>
    %2 = tosa.sub %1, %1 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %r_3 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.reshape %2, %r_3 : (tensor<i32>, !tosa.shape<2>) -> tensor<1x1xi32>
    %4 = tosa.reduce_any %arg1 {axis = 1 : i32} : (tensor<91x63x29xi1>) -> tensor<91x1x29xi1>
    %5 = tosa.intdiv %3, %3 : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
    %6 = tosa.ceil %arg2 : (tensor<25x40x1x79xf32>) -> tensor<25x40x1x79xf32>
    %7 = tosa.reduce_all %4 {axis = 0 : i32} : (tensor<91x1x29xi1>) -> tensor<1x1x29xi1>
    %8 = tosa.bitwise_and %7, %7 : (tensor<1x1x29xi1>, tensor<1x1x29xi1>) -> tensor<1x1x29xi1>
    return %5, %6, %8 : tensor<1x1xi32>, tensor<25x40x1x79xf32>, tensor<1x1x29xi1>
  }
}
