module {
  func.func @main(%arg0: tensor<79xi1>, %arg1: tensor<63x21x26x65x67xf32>, %arg2: tensor<96xi32>, %arg3: tensor<96xi32>) -> (tensor<63x21x26x65x67xf32>, tensor<96xi32>, tensor<2xi1>) {
    %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<79xi1>) -> tensor<1xi1>
    %1 = tosa.rsqrt %arg1 : (tensor<63x21x26x65x67xf32>) -> tensor<63x21x26x65x67xf32>
    %2 = tosa.bitwise_or %0, %0 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %3 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %4 = tosa.transpose %2 {perms = array<i32: 0>} : (tensor<1xi1>) -> tensor<1xi1>
    %5 = tosa.reverse %4 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %t_6 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %6 = tosa.tile %5, %t_6 : (tensor<1xi1>, !tosa.shape<1>) -> tensor<2xi1>
    %7 = tosa.intdiv %arg2, %arg3 : (tensor<96xi32>, tensor<96xi32>) -> tensor<96xi32>
    %8 = tosa.reverse %6 {axis = 0 : i32} : (tensor<2xi1>) -> tensor<2xi1>
    return %1, %7, %8 : tensor<63x21x26x65x67xf32>, tensor<96xi32>, tensor<2xi1>
  }
}
