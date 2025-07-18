module {
  func.func @main(%arg0: tensor<5x36x56x40xf32>, %arg1: tensor<i8>, %arg2: tensor<49x42xi1>, %arg3: tensor<1x1xi1>) -> (tensor<22400x1xf32>, tensor<i8>, tensor<49x42xi1>) {
    %0 = tosa.exp %arg0 : (tensor<5x36x56x40xf32>) -> tensor<5x36x56x40xf32>
    %1 = tosa.log %0 : (tensor<5x36x56x40xf32>) -> tensor<5x36x56x40xf32>
    %2 = tosa.ceil %1 : (tensor<5x36x56x40xf32>) -> tensor<5x36x56x40xf32>
    %3 = tosa.sub %2, %2 : (tensor<5x36x56x40xf32>, tensor<5x36x56x40xf32>) -> tensor<5x36x56x40xf32>
    %4 = tosa.floor %3 : (tensor<5x36x56x40xf32>) -> tensor<5x36x56x40xf32>
    %5 = tosa.reciprocal %4 : (tensor<5x36x56x40xf32>) -> tensor<5x36x56x40xf32>
    %r_6 = tosa.const_shape {values = dense<[ 11200, 36 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %6 = tosa.reshape %5, %r_6 : (tensor<5x36x56x40xf32>, !tosa.shape<2>) -> tensor<11200x36xf32>
    %7 = tosa.ceil %6 : (tensor<11200x36xf32>) -> tensor<11200x36xf32>
    %8 = tosa.pow %7, %7 : (tensor<11200x36xf32>, tensor<11200x36xf32>) -> tensor<11200x36xf32>
    %9 = tosa.reduce_min %8 {axis = 1 : i32} : (tensor<11200x36xf32>) -> tensor<11200x1xf32>
    %10 = tosa.sub %9, %9 : (tensor<11200x1xf32>, tensor<11200x1xf32>) -> tensor<11200x1xf32>
    %11 = tosa.concat %10, %9 {axis = 0 : i32} : (tensor<11200x1xf32>, tensor<11200x1xf32>) -> tensor<22400x1xf32>
    %12 = tosa.clz %arg1 : (tensor<i8>) -> tensor<i8>
    %13 = tosa.bitwise_or %12, %12 : (tensor<i8>, tensor<i8>) -> tensor<i8>
    %14 = tosa.logical_or %arg2, %arg3 : (tensor<49x42xi1>, tensor<1x1xi1>) -> tensor<49x42xi1>
    %15 = tosa.reverse %14 {axis = 1 : i32} : (tensor<49x42xi1>) -> tensor<49x42xi1>
    return %11, %13, %15 : tensor<22400x1xf32>, tensor<i8>, tensor<49x42xi1>
  }
}
