module {
  func.func @main(%arg0: tensor<62x18x27x63x21xf32>, %arg1: tensor<100xi1>) -> (tensor<300xi1>, tensor<62x18x27x63x21xi1>, tensor<i32>) {
    %0 = tosa.reciprocal %arg0 : (tensor<62x18x27x63x21xf32>) -> tensor<62x18x27x63x21xf32>
    %1 = tosa.ceil %0 : (tensor<62x18x27x63x21xf32>) -> tensor<62x18x27x63x21xf32>
    %2 = tosa.add %1, %0 : (tensor<62x18x27x63x21xf32>, tensor<62x18x27x63x21xf32>) -> tensor<62x18x27x63x21xf32>
    %3 = tosa.floor %2 : (tensor<62x18x27x63x21xf32>) -> tensor<62x18x27x63x21xf32>
    %t_4 = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %4 = tosa.tile %arg1, %t_4 : (tensor<100xi1>, !tosa.shape<1>) -> tensor<300xi1>
    %5 = tosa.exp %3 : (tensor<62x18x27x63x21xf32>) -> tensor<62x18x27x63x21xf32>
    %6 = tosa.clz %4 : (tensor<300xi1>) -> tensor<300xi1>
    %7 = tosa.abs %4 : (tensor<300xi1>) -> tensor<300xi1>
    %8 = tosa.equal %5, %0 : (tensor<62x18x27x63x21xf32>, tensor<62x18x27x63x21xf32>) -> tensor<62x18x27x63x21xi1>
    %t_9 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %9 = tosa.tile %7, %t_9 : (tensor<300xi1>, !tosa.shape<1>) -> tensor<600xi1>
    %10 = tosa.bitwise_and %9, %9 : (tensor<600xi1>, tensor<600xi1>) -> tensor<600xi1>
    %11 = tosa.argmax %10 {axis = 0 : i32} : (tensor<600xi1>) -> tensor<i32>
    return %6, %8, %11 : tensor<300xi1>, tensor<62x18x27x63x21xi1>, tensor<i32>
  }
}
