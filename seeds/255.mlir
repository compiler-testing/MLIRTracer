module {
  func.func @main(%arg0: tensor<88x30x5xi1>, %arg1: tensor<88x5x66xi1>, %arg2: tensor<86x64xi8>, %arg3: tensor<1x64xi8>, %arg4: tensor<22x33x24x47x4xf32>, %arg5: tensor<52x57x89x77x70x52xi32>, %arg6: tensor<1x1x1x77x1x1xi32>) -> (tensor<88x30x66xi1>, tensor<22x33x24x47x4xi1>, tensor<1x64xi1>, tensor<22x33x24x47x4xi1>, tensor<22x33x24x47x4xf32>, tensor<1x3102x1056xf32>, tensor<86x64xi1>, tensor<1x3102x1056xf32>, tensor<1x3102x1056xf32>, tensor<52x57x89x77x70x52xi32>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<88x30x5xi1>, tensor<88x5x66xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<88x30x66xi1>
    %1 = tosa.reverse %0 {axis = 0 : i32} : (tensor<88x30x66xi1>) -> tensor<88x30x66xi1>
    %2 = tosa.greater_equal %arg2, %arg3 : (tensor<86x64xi8>, tensor<1x64xi8>) -> tensor<86x64xi1>
    %3 = tosa.bitwise_and %2, %2 : (tensor<86x64xi1>, tensor<86x64xi1>) -> tensor<86x64xi1>
    %4 = tosa.tanh %arg4 : (tensor<22x33x24x47x4xf32>) -> tensor<22x33x24x47x4xf32>
    %5 = tosa.greater_equal %4, %4 : (tensor<22x33x24x47x4xf32>, tensor<22x33x24x47x4xf32>) -> tensor<22x33x24x47x4xi1>
    %6 = tosa.reverse %3 {axis = 0 : i32} : (tensor<86x64xi1>) -> tensor<86x64xi1>
    %7 = tosa.reduce_all %6 {axis = 0 : i32} : (tensor<86x64xi1>) -> tensor<1x64xi1>
    %8 = tosa.equal %4, %4 : (tensor<22x33x24x47x4xf32>, tensor<22x33x24x47x4xf32>) -> tensor<22x33x24x47x4xi1>
    %9 = tosa.sigmoid %4 : (tensor<22x33x24x47x4xf32>) -> tensor<22x33x24x47x4xf32>
    %10 = tosa.log %4 : (tensor<22x33x24x47x4xf32>) -> tensor<22x33x24x47x4xf32>
    %r_11 = tosa.const_shape {values = dense<[ 1, 3102, 1056 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %11 = tosa.reshape %10, %r_11 : (tensor<22x33x24x47x4xf32>, !tosa.shape<3>) -> tensor<1x3102x1056xf32>
    %12 = tosa.log %11 : (tensor<1x3102x1056xf32>) -> tensor<1x3102x1056xf32>
    %13 = tosa.reverse %11 {axis = 1 : i32} : (tensor<1x3102x1056xf32>) -> tensor<1x3102x1056xf32>
    %14 = tosa.abs %12 : (tensor<1x3102x1056xf32>) -> tensor<1x3102x1056xf32>
    %15 = tosa.ceil %13 : (tensor<1x3102x1056xf32>) -> tensor<1x3102x1056xf32>
    %16 = tosa.sub %12, %15 : (tensor<1x3102x1056xf32>, tensor<1x3102x1056xf32>) -> tensor<1x3102x1056xf32>
    %17 = tosa.logical_right_shift %6, %3 : (tensor<86x64xi1>, tensor<86x64xi1>) -> tensor<86x64xi1>
    %18 = tosa.minimum %12, %16 : (tensor<1x3102x1056xf32>, tensor<1x3102x1056xf32>) -> tensor<1x3102x1056xf32>
    %19 = tosa.intdiv %arg5, %arg6 : (tensor<52x57x89x77x70x52xi32>, tensor<1x1x1x77x1x1xi32>) -> tensor<52x57x89x77x70x52xi32>
    %20 = tosa.arithmetic_right_shift %19, %19 {round = false} : (tensor<52x57x89x77x70x52xi32>, tensor<52x57x89x77x70x52xi32>) -> tensor<52x57x89x77x70x52xi32>
    %21 = tosa.sigmoid %18 : (tensor<1x3102x1056xf32>) -> tensor<1x3102x1056xf32>
    %22 = tosa.abs %18 : (tensor<1x3102x1056xf32>) -> tensor<1x3102x1056xf32>
    %23 = tosa.bitwise_xor %19, %20 : (tensor<52x57x89x77x70x52xi32>, tensor<52x57x89x77x70x52xi32>) -> tensor<52x57x89x77x70x52xi32>
    return %1, %5, %7, %8, %9, %14, %17, %21, %22, %23 : tensor<88x30x66xi1>, tensor<22x33x24x47x4xi1>, tensor<1x64xi1>, tensor<22x33x24x47x4xi1>, tensor<22x33x24x47x4xf32>, tensor<1x3102x1056xf32>, tensor<86x64xi1>, tensor<1x3102x1056xf32>, tensor<1x3102x1056xf32>, tensor<52x57x89x77x70x52xi32>
  }
}
