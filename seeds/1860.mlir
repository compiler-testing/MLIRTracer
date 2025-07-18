module {
  func.func @main(%arg0: tensor<16xf32>) -> (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1x1x1xi32>) {
    %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<16xf32>) -> tensor<1xf32>
    %1 = tosa.reduce_sum %0 {axis = 0 : i32} : (tensor<1xf32>) -> tensor<1xf32>
    %2 = tosa.maximum %1, %0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %3 = tosa.argmax %2 {axis = 0 : i32} : (tensor<1xf32>) -> tensor<i32>
    %4 = tosa.maximum %2, %1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %5 = tosa.abs %3 : (tensor<i32>) -> tensor<i32>
    %6 = tosa.tanh %2 : (tensor<1xf32>) -> tensor<1xf32>
    %r_7 = tosa.const_shape {values = dense<[ 1, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %7 = tosa.reshape %5, %r_7 : (tensor<i32>, !tosa.shape<3>) -> tensor<1x1x1xi32>
    %8 = tosa.intdiv %7, %7 : (tensor<1x1x1xi32>, tensor<1x1x1xi32>) -> tensor<1x1x1xi32>
    %9 = tosa.reciprocal %2 : (tensor<1xf32>) -> tensor<1xf32>
    %10 = tosa.reciprocal %6 : (tensor<1xf32>) -> tensor<1xf32>
    %11 = tosa.bitwise_and %8, %7 : (tensor<1x1x1xi32>, tensor<1x1x1xi32>) -> tensor<1x1x1xi32>
    %12 = tosa.sub %11, %7 : (tensor<1x1x1xi32>, tensor<1x1x1xi32>) -> tensor<1x1x1xi32>
    %13 = tosa.reduce_product %12 {axis = 0 : i32} : (tensor<1x1x1xi32>) -> tensor<1x1x1xi32>
    %14 = tosa.reduce_min %13 {axis = 1 : i32} : (tensor<1x1x1xi32>) -> tensor<1x1x1xi32>
    %15 = tosa.bitwise_or %14, %11 : (tensor<1x1x1xi32>, tensor<1x1x1xi32>) -> tensor<1x1x1xi32>
    %16 = tosa.intdiv %15, %12 : (tensor<1x1x1xi32>, tensor<1x1x1xi32>) -> tensor<1x1x1xi32>
    return %4, %9, %10, %16 : tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1x1x1xi32>
  }
}
