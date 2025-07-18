module {
  func.func @main(%arg0: tensor<25xi16>, %arg1: tensor<97x40x50x26x20x31xf32>, %arg2: tensor<97x40x1x1x20x31xf32>) -> (tensor<i32>, tensor<25xi16>, tensor<97x40x50x26x20x31xf32>, tensor<1xi16>, tensor<1xi16>, tensor<97x40x50x26x20x31xf32>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<25xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<25xi16>
    %1 = tosa.pow %arg1, %arg2 : (tensor<97x40x50x26x20x31xf32>, tensor<97x40x1x1x20x31xf32>) -> tensor<97x40x50x26x20x31xf32>
    %2 = tosa.rsqrt %1 : (tensor<97x40x50x26x20x31xf32>) -> tensor<97x40x50x26x20x31xf32>
    %3 = tosa.sigmoid %2 : (tensor<97x40x50x26x20x31xf32>) -> tensor<97x40x50x26x20x31xf32>
    %4 = tosa.identity %3 : (tensor<97x40x50x26x20x31xf32>) -> tensor<97x40x50x26x20x31xf32>
    %5 = tosa.reduce_product %0 {axis = 0 : i32} : (tensor<25xi16>) -> tensor<1xi16>
    %6 = tosa.logical_left_shift %0, %0 : (tensor<25xi16>, tensor<25xi16>) -> tensor<25xi16>
    %7 = tosa.argmax %6 {axis = 0 : i32} : (tensor<25xi16>) -> tensor<i32>
    %8 = tosa.log %1 : (tensor<97x40x50x26x20x31xf32>) -> tensor<97x40x50x26x20x31xf32>
    %9 = tosa.tanh %4 : (tensor<97x40x50x26x20x31xf32>) -> tensor<97x40x50x26x20x31xf32>
    %10 = tosa.reverse %5 {axis = 0 : i32} : (tensor<1xi16>) -> tensor<1xi16>
    %11 = tosa.pow %8, %3 : (tensor<97x40x50x26x20x31xf32>, tensor<97x40x50x26x20x31xf32>) -> tensor<97x40x50x26x20x31xf32>
    %12 = tosa.logical_left_shift %0, %6 : (tensor<25xi16>, tensor<25xi16>) -> tensor<25xi16>
    %13 = tosa.ceil %9 : (tensor<97x40x50x26x20x31xf32>) -> tensor<97x40x50x26x20x31xf32>
    %14 = tosa.add %13, %11 : (tensor<97x40x50x26x20x31xf32>, tensor<97x40x50x26x20x31xf32>) -> tensor<97x40x50x26x20x31xf32>
    %15 = tosa.reduce_product %6 {axis = 0 : i32} : (tensor<25xi16>) -> tensor<1xi16>
    %16 = tosa.ceil %11 : (tensor<97x40x50x26x20x31xf32>) -> tensor<97x40x50x26x20x31xf32>
    %17 = tosa.logical_right_shift %5, %10 : (tensor<1xi16>, tensor<1xi16>) -> tensor<1xi16>
    %18 = tosa.exp %16 : (tensor<97x40x50x26x20x31xf32>) -> tensor<97x40x50x26x20x31xf32>
    return %7, %12, %14, %15, %17, %18 : tensor<i32>, tensor<25xi16>, tensor<97x40x50x26x20x31xf32>, tensor<1xi16>, tensor<1xi16>, tensor<97x40x50x26x20x31xf32>
  }
}
