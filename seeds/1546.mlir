module {
  func.func @main(%arg0: tensor<68x84x10x1xf32>, %arg1: tensor<1x1x1x1xf32>, %arg2: tensor<67x100x13x9x44xi8>, %arg3: tensor<67x1x13x9x44xi8>, %arg4: tensor<17x32x40xi32>, %arg5: tensor<1x32x40xi32>, %arg6: tensor<40x84x24xi1>) -> (tensor<67x100x13x9x44xi8>, tensor<68x1x10x1xf32>, tensor<40x1x24xi1>, tensor<68x84x10x1xf32>, tensor<17x32x40xi32>, tensor<17x32x40xi1>, tensor<17x32x40xi32>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<68x84x10x1xf32>, tensor<1x1x1x1xf32>) -> tensor<68x84x10x1xf32>
    %in_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.negate %0, %in_zp_1, %out_zp_1 : (tensor<68x84x10x1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<68x84x10x1xf32>
    %2 = tosa.reduce_product %1 {axis = 1 : i32} : (tensor<68x84x10x1xf32>) -> tensor<68x1x10x1xf32>
    %3 = tosa.bitwise_or %arg2, %arg3 : (tensor<67x100x13x9x44xi8>, tensor<67x1x13x9x44xi8>) -> tensor<67x100x13x9x44xi8>
    %4 = tosa.add %3, %3 : (tensor<67x100x13x9x44xi8>, tensor<67x100x13x9x44xi8>) -> tensor<67x100x13x9x44xi8>
    %5 = tosa.intdiv %arg4, %arg5 : (tensor<17x32x40xi32>, tensor<1x32x40xi32>) -> tensor<17x32x40xi32>
    %6 = tosa.pow %2, %2 : (tensor<68x1x10x1xf32>, tensor<68x1x10x1xf32>) -> tensor<68x1x10x1xf32>
    %7 = tosa.reduce_all %arg6 {axis = 1 : i32} : (tensor<40x84x24xi1>) -> tensor<40x1x24xi1>
    %8 = tosa.bitwise_xor %5, %5 : (tensor<17x32x40xi32>, tensor<17x32x40xi32>) -> tensor<17x32x40xi32>
    %9 = tosa.intdiv %5, %5 : (tensor<17x32x40xi32>, tensor<17x32x40xi32>) -> tensor<17x32x40xi32>
    %10 = tosa.greater_equal %9, %9 : (tensor<17x32x40xi32>, tensor<17x32x40xi32>) -> tensor<17x32x40xi1>
    %11 = tosa.pow %0, %0 : (tensor<68x84x10x1xf32>, tensor<68x84x10x1xf32>) -> tensor<68x84x10x1xf32>
    %12 = tosa.minimum %9, %8 : (tensor<17x32x40xi32>, tensor<17x32x40xi32>) -> tensor<17x32x40xi32>
    %13 = tosa.bitwise_xor %10, %10 : (tensor<17x32x40xi1>, tensor<17x32x40xi1>) -> tensor<17x32x40xi1>
    %14 = tosa.bitwise_and %8, %5 : (tensor<17x32x40xi32>, tensor<17x32x40xi32>) -> tensor<17x32x40xi32>
    return %4, %6, %7, %11, %12, %13, %14 : tensor<67x100x13x9x44xi8>, tensor<68x1x10x1xf32>, tensor<40x1x24xi1>, tensor<68x84x10x1xf32>, tensor<17x32x40xi32>, tensor<17x32x40xi1>, tensor<17x32x40xi32>
  }
}
