module {
  func.func @main(%arg0: tensor<76x67x76x51x26x59xf32>, %arg1: tensor<73x8xi16>) -> (tensor<76x67x76x51x26x59xf32>, tensor<152x67x76x51x26x59xi1>, tensor<76x67x76x51x26x59xf32>, tensor<8xi1>, tensor<76x67x76x51x26x59xf32>, tensor<76x67x76x51x26x59xf32>, tensor<8xi32>) {
    %0 = tosa.ceil %arg0 : (tensor<76x67x76x51x26x59xf32>) -> tensor<76x67x76x51x26x59xf32>
    %1 = tosa.equal %0, %0 : (tensor<76x67x76x51x26x59xf32>, tensor<76x67x76x51x26x59xf32>) -> tensor<76x67x76x51x26x59xi1>
    %2 = tosa.concat %1, %1 {axis = 0 : i32} : (tensor<76x67x76x51x26x59xi1>, tensor<76x67x76x51x26x59xi1>) -> tensor<152x67x76x51x26x59xi1>
    %3 = tosa.floor %0 : (tensor<76x67x76x51x26x59xf32>) -> tensor<76x67x76x51x26x59xf32>
    %in_zp_4 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_4 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %4 = tosa.negate %2, %in_zp_4, %out_zp_4 : (tensor<152x67x76x51x26x59xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<152x67x76x51x26x59xi1>
    %5 = tosa.ceil %0 : (tensor<76x67x76x51x26x59xf32>) -> tensor<76x67x76x51x26x59xf32>
    %6 = tosa.arithmetic_right_shift %4, %4 {round = true} : (tensor<152x67x76x51x26x59xi1>, tensor<152x67x76x51x26x59xi1>) -> tensor<152x67x76x51x26x59xi1>
    %7 = tosa.bitwise_or %6, %6 : (tensor<152x67x76x51x26x59xi1>, tensor<152x67x76x51x26x59xi1>) -> tensor<152x67x76x51x26x59xi1>
    %8 = tosa.clz %7 : (tensor<152x67x76x51x26x59xi1>) -> tensor<152x67x76x51x26x59xi1>
    %9 = tosa.argmax %arg1 {axis = 0 : i32} : (tensor<73x8xi16>) -> tensor<8xi32>
    %10 = tosa.ceil %3 : (tensor<76x67x76x51x26x59xf32>) -> tensor<76x67x76x51x26x59xf32>
    %11 = tosa.tanh %5 : (tensor<76x67x76x51x26x59xf32>) -> tensor<76x67x76x51x26x59xf32>
    %12 = tosa.maximum %9, %9 : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
    %13 = tosa.logical_not %8 : (tensor<152x67x76x51x26x59xi1>) -> tensor<152x67x76x51x26x59xi1>
    %14 = tosa.clamp %12 {min_val = -49 : i32, max_val = 75 : i32} : (tensor<8xi32>) -> tensor<8xi32>
    %15 = tosa.reverse %9 {axis = 0 : i32} : (tensor<8xi32>) -> tensor<8xi32>
    %16 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %17 = tosa.transpose %14 {perms = array<i32: 0>} : (tensor<8xi32>) -> tensor<8xi32>
    %18 = tosa.intdiv %15, %9 : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
    %19 = tosa.floor %5 : (tensor<76x67x76x51x26x59xf32>) -> tensor<76x67x76x51x26x59xf32>
    %20 = tosa.equal %17, %15 : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi1>
    %21 = tosa.tanh %11 : (tensor<76x67x76x51x26x59xf32>) -> tensor<76x67x76x51x26x59xf32>
    %22 = tosa.exp %0 : (tensor<76x67x76x51x26x59xf32>) -> tensor<76x67x76x51x26x59xf32>
    %23 = tosa.intdiv %18, %18 : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
    return %10, %13, %19, %20, %21, %22, %23 : tensor<76x67x76x51x26x59xf32>, tensor<152x67x76x51x26x59xi1>, tensor<76x67x76x51x26x59xf32>, tensor<8xi1>, tensor<76x67x76x51x26x59xf32>, tensor<76x67x76x51x26x59xf32>, tensor<8xi32>
  }
}
