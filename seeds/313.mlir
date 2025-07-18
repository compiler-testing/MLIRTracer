module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<14x1x28x70xf32>) -> (tensor<1x1x1xi1>, tensor<1x1x1xi1>, tensor<14x1x28x70xf32>, tensor<i1>, tensor<i32>, tensor<i32>, tensor<28x3x56x210xf32>, tensor<14x1x28x70xf32>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %r_1 = tosa.const_shape {values = dense<[ 1, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.reshape %0, %r_1 : (tensor<i1>, !tosa.shape<3>) -> tensor<1x1x1xi1>
    %2 = tosa.clz %1 : (tensor<1x1x1xi1>) -> tensor<1x1x1xi1>
    %3 = tosa.intdiv %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %4 = tosa.reduce_min %2 {axis = 1 : i32} : (tensor<1x1x1xi1>) -> tensor<1x1x1xi1>
    %5 = tosa.greater_equal %3, %3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %6 = tosa.clz %3 : (tensor<i32>) -> tensor<i32>
    %7 = tosa.rsqrt %arg4 : (tensor<14x1x28x70xf32>) -> tensor<14x1x28x70xf32>
    %8 = tosa.logical_or %5, %5 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %9 = tosa.reduce_any %2 {axis = 1 : i32} : (tensor<1x1x1xi1>) -> tensor<1x1x1xi1>
    %10 = tosa.rsqrt %7 : (tensor<14x1x28x70xf32>) -> tensor<14x1x28x70xf32>
    %11 = tosa.reduce_product %10 {axis = 1 : i32} : (tensor<14x1x28x70xf32>) -> tensor<14x1x28x70xf32>
    %12 = tosa.floor %11 : (tensor<14x1x28x70xf32>) -> tensor<14x1x28x70xf32>
    %13 = tosa.equal %6, %6 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %14 = tosa.logical_and %13, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %15 = tosa.bitwise_or %14, %8 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %16 = tosa.bitwise_xor %6, %6 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %t_17 = tosa.const_shape {values = dense<[ 2, 3, 2, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %17 = tosa.tile %10, %t_17 : (tensor<14x1x28x70xf32>, !tosa.shape<4>) -> tensor<28x3x56x210xf32>
    %18 = tosa.bitwise_or %6, %6 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %19 = tosa.floor %17 : (tensor<28x3x56x210xf32>) -> tensor<28x3x56x210xf32>
    %20 = tosa.sigmoid %7 : (tensor<14x1x28x70xf32>) -> tensor<14x1x28x70xf32>
    return %4, %9, %12, %15, %16, %18, %19, %20 : tensor<1x1x1xi1>, tensor<1x1x1xi1>, tensor<14x1x28x70xf32>, tensor<i1>, tensor<i32>, tensor<i32>, tensor<28x3x56x210xf32>, tensor<14x1x28x70xf32>
  }
}
