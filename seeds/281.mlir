module {
  func.func @main(%arg0: tensor<41x51x41x72x91xi64>, %arg1: tensor<1x51x41x1x91xi64>, %arg2: tensor<22x38x37x51x35xf32>, %arg3: tensor<47xf32>, %arg4: tensor<65x71x70xi1>, %arg5: tensor<1x1x1xi1>, %arg6: tensor<24x1x94xi32>, %arg7: tensor<1x1x94xi32>) -> (tensor<41x51x41x72x91xi64>, tensor<24x1x94xi32>, tensor<2xf32>, tensor<2275x2x71x1xi1>, tensor<22x38x37x51x35xf32>, tensor<65x71x70xi1>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<41x51x41x72x91xi64>, tensor<1x51x41x1x91xi64>) -> tensor<41x51x41x72x91xi64>
    %1 = tosa.reciprocal %arg2 : (tensor<22x38x37x51x35xf32>) -> tensor<22x38x37x51x35xf32>
    %2 = tosa.abs %1 : (tensor<22x38x37x51x35xf32>) -> tensor<22x38x37x51x35xf32>
    %3 = tosa.log %2 : (tensor<22x38x37x51x35xf32>) -> tensor<22x38x37x51x35xf32>
    %4 = tosa.reduce_sum %arg3 {axis = 0 : i32} : (tensor<47xf32>) -> tensor<1xf32>
    %5 = tosa.bitwise_xor %0, %0 : (tensor<41x51x41x72x91xi64>, tensor<41x51x41x72x91xi64>) -> tensor<41x51x41x72x91xi64>
    %6 = tosa.abs %3 : (tensor<22x38x37x51x35xf32>) -> tensor<22x38x37x51x35xf32>
    %7 = tosa.pow %4, %4 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %8 = tosa.logical_and %arg4, %arg5 : (tensor<65x71x70xi1>, tensor<1x1x1xi1>) -> tensor<65x71x70xi1>
    %9 = tosa.concat %4, %7 {axis = 0 : i32} : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %in_zp_10 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_10 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %10 = tosa.negate %8, %in_zp_10, %out_zp_10 : (tensor<65x71x70xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<65x71x70xi1>
    %11 = tosa.intdiv %arg6, %arg7 : (tensor<24x1x94xi32>, tensor<1x1x94xi32>) -> tensor<24x1x94xi32>
    %12 = tosa.reciprocal %9 : (tensor<2xf32>) -> tensor<2xf32>
    %r_13 = tosa.const_shape {values = dense<[ 2275, 2, 71, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %13 = tosa.reshape %10, %r_13 : (tensor<65x71x70xi1>, !tosa.shape<4>) -> tensor<2275x2x71x1xi1>
    %14 = tosa.sigmoid %6 : (tensor<22x38x37x51x35xf32>) -> tensor<22x38x37x51x35xf32>
    %15 = tosa.reciprocal %14 : (tensor<22x38x37x51x35xf32>) -> tensor<22x38x37x51x35xf32>
    %16 = tosa.sub %10, %10 : (tensor<65x71x70xi1>, tensor<65x71x70xi1>) -> tensor<65x71x70xi1>
    return %5, %11, %12, %13, %15, %16 : tensor<41x51x41x72x91xi64>, tensor<24x1x94xi32>, tensor<2xf32>, tensor<2275x2x71x1xi1>, tensor<22x38x37x51x35xf32>, tensor<65x71x70xi1>
  }
}
