module {
  func.func @main(%arg0: tensor<56x80x28x18x62x49xf32>, %arg1: tensor<72x51xi16>) -> (tensor<56x80x56x18x62x49xi1>, tensor<10x1xi16>, tensor<56x80x28x18x62x49xf32>, tensor<10x10xi16>) {
    %0 = tosa.reciprocal %arg0 : (tensor<56x80x28x18x62x49xf32>) -> tensor<56x80x28x18x62x49xf32>
    %1 = tosa.reciprocal %0 : (tensor<56x80x28x18x62x49xf32>) -> tensor<56x80x28x18x62x49xf32>
    %2 = tosa.rsqrt %1 : (tensor<56x80x28x18x62x49xf32>) -> tensor<56x80x28x18x62x49xf32>
    %3 = tosa.reverse %arg1 {axis = 1 : i32} : (tensor<72x51xi16>) -> tensor<72x51xi16>
    %4 = "tosa.const"() {values = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    %5 = tosa.transpose %3 {perms = array<i32: 0, 1>} : (tensor<72x51xi16>) -> tensor<72x51xi16>
    %6 = tosa.tanh %2 : (tensor<56x80x28x18x62x49xf32>) -> tensor<56x80x28x18x62x49xf32>
    %7 = tosa.minimum %6, %0 : (tensor<56x80x28x18x62x49xf32>, tensor<56x80x28x18x62x49xf32>) -> tensor<56x80x28x18x62x49xf32>
    %8 = tosa.floor %7 : (tensor<56x80x28x18x62x49xf32>) -> tensor<56x80x28x18x62x49xf32>
    %s_9_start = tosa.const_shape {values = dense<[ 62, 41 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_9_size = tosa.const_shape {values = dense<[ 10, 10 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %9 = tosa.slice %5, %s_9_start, %s_9_size : (tensor<72x51xi16>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<10x10xi16>
    %10 = tosa.add %9, %9 : (tensor<10x10xi16>, tensor<10x10xi16>) -> tensor<10x10xi16>
    %11 = tosa.concat %8, %1 {axis = 2 : i32} : (tensor<56x80x28x18x62x49xf32>, tensor<56x80x28x18x62x49xf32>) -> tensor<56x80x56x18x62x49xf32>
    %12 = tosa.greater_equal %11, %11 : (tensor<56x80x56x18x62x49xf32>, tensor<56x80x56x18x62x49xf32>) -> tensor<56x80x56x18x62x49xi1>
    %13 = tosa.reduce_product %10 {axis = 1 : i32} : (tensor<10x10xi16>) -> tensor<10x1xi16>
    %14 = tosa.ceil %2 : (tensor<56x80x28x18x62x49xf32>) -> tensor<56x80x28x18x62x49xf32>
    %15 = tosa.bitwise_not %10 : (tensor<10x10xi16>) -> tensor<10x10xi16>
    return %12, %13, %14, %15 : tensor<56x80x56x18x62x49xi1>, tensor<10x1xi16>, tensor<56x80x28x18x62x49xf32>, tensor<10x10xi16>
  }
}
