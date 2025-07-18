module {
  func.func @main(%arg0: tensor<100x28x63xf32>, %arg1: tensor<100x1x63xf32>, %arg2: tensor<89x47x74x56x62xi1>, %arg3: tensor<1x47x1x56x62xi1>) -> (tensor<56x63xi1>, tensor<210x14xi1>, tensor<56x63xi32>, tensor<100x28x63xf32>, tensor<100x28x63xf32>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<100x28x63xf32>, tensor<100x1x63xf32>) -> tensor<100x28x63xf32>
    %1 = tosa.pow %0, %0 : (tensor<100x28x63xf32>, tensor<100x28x63xf32>) -> tensor<100x28x63xf32>
    %2 = tosa.minimum %1, %1 : (tensor<100x28x63xf32>, tensor<100x28x63xf32>) -> tensor<100x28x63xf32>
    %3 = tosa.argmax %2 {axis = 0 : i32} : (tensor<100x28x63xf32>) -> tensor<28x63xi32>
    %4 = tosa.concat %3, %3 {axis = 0 : i32} : (tensor<28x63xi32>, tensor<28x63xi32>) -> tensor<56x63xi32>
    %5 = tosa.logical_or %arg2, %arg3 : (tensor<89x47x74x56x62xi1>, tensor<1x47x1x56x62xi1>) -> tensor<89x47x74x56x62xi1>
    %s_6_start = tosa.const_shape {values = dense<[ 73, 45, 21, 50, 9 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_6_size = tosa.const_shape {values = dense<[ 7, 2, 7, 6, 5 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %6 = tosa.slice %5, %s_6_start, %s_6_size : (tensor<89x47x74x56x62xi1>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<7x2x7x6x5xi1>
    %7 = tosa.clamp %4 {min_val = -11 : i32, max_val = 109 : i32} : (tensor<56x63xi32>) -> tensor<56x63xi32>
    %8 = tosa.greater_equal %7, %7 : (tensor<56x63xi32>, tensor<56x63xi32>) -> tensor<56x63xi1>
    %9 = tosa.logical_or %8, %8 : (tensor<56x63xi1>, tensor<56x63xi1>) -> tensor<56x63xi1>
    %10 = tosa.tanh %2 : (tensor<100x28x63xf32>) -> tensor<100x28x63xf32>
    %r_11 = tosa.const_shape {values = dense<[ 210, 14 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %11 = tosa.reshape %6, %r_11 : (tensor<7x2x7x6x5xi1>, !tosa.shape<2>) -> tensor<210x14xi1>
    %12 = tosa.intdiv %4, %4 : (tensor<56x63xi32>, tensor<56x63xi32>) -> tensor<56x63xi32>
    %13 = tosa.pow %10, %0 : (tensor<100x28x63xf32>, tensor<100x28x63xf32>) -> tensor<100x28x63xf32>
    %14 = tosa.ceil %2 : (tensor<100x28x63xf32>) -> tensor<100x28x63xf32>
    return %9, %11, %12, %13, %14 : tensor<56x63xi1>, tensor<210x14xi1>, tensor<56x63xi32>, tensor<100x28x63xf32>, tensor<100x28x63xf32>
  }
}
