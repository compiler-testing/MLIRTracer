module {
  func.func @main(%arg0: tensor<i8>, %arg1: tensor<i8>, %arg2: tensor<92xf32>) -> (tensor<i1>, tensor<1xi1>, tensor<1xf32>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<i8>, tensor<i8>) -> tensor<i8>
    %1 = tosa.greater_equal %0, %0 : (tensor<i8>, tensor<i8>) -> tensor<i1>
    %2 = tosa.reduce_product %arg2 {axis = 0 : i32} : (tensor<92xf32>) -> tensor<1xf32>
    %3 = tosa.logical_left_shift %1, %1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = tosa.greater %2, %2 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %5 = tosa.clz %4 : (tensor<1xi1>) -> tensor<1xi1>
    %6 = tosa.bitwise_xor %5, %5 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %7 = tosa.sigmoid %2 : (tensor<1xf32>) -> tensor<1xf32>
    return %3, %6, %7 : tensor<i1>, tensor<1xi1>, tensor<1xf32>
  }
}
