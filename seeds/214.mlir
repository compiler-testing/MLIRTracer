module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<80x38x98x72xi1>, %arg3: tensor<94x37x57x1x20x52xi64>, %arg4: tensor<1x37x57x1x20x52xi64>, %arg5: tensor<32x28x35x91x53xi8>, %arg6: tensor<32x28x35x1x53xi8>, %arg7: tensor<10x91x35x97xf32>) -> (tensor<80x1x98x72xi1>, tensor<1x1x2x103087920xi1>, tensor<i1>, tensor<94x37x57x1x20x52xi1>, tensor<32x28x35x91x53xi1>, tensor<1x1x35x97xf32>, tensor<10x91x35xi32>, tensor<10x91x35x97xf32>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %1 = tosa.reduce_any %arg2 {axis = 1 : i32} : (tensor<80x38x98x72xi1>) -> tensor<80x1x98x72xi1>
    %2 = tosa.logical_left_shift %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %3 = tosa.logical_right_shift %1, %1 : (tensor<80x1x98x72xi1>, tensor<80x1x98x72xi1>) -> tensor<80x1x98x72xi1>
    %4 = tosa.bitwise_xor %3, %1 : (tensor<80x1x98x72xi1>, tensor<80x1x98x72xi1>) -> tensor<80x1x98x72xi1>
    %5 = tosa.equal %arg3, %arg4 : (tensor<94x37x57x1x20x52xi64>, tensor<1x37x57x1x20x52xi64>) -> tensor<94x37x57x1x20x52xi1>
    %r_6 = tosa.const_shape {values = dense<[ 1, 1, 2, 103087920 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %6 = tosa.reshape %5, %r_6 : (tensor<94x37x57x1x20x52xi1>, !tosa.shape<4>) -> tensor<1x1x2x103087920xi1>
    %7 = tosa.arithmetic_right_shift %2, %2 {round = true} : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %8 = tosa.logical_right_shift %5, %5 : (tensor<94x37x57x1x20x52xi1>, tensor<94x37x57x1x20x52xi1>) -> tensor<94x37x57x1x20x52xi1>
    %9 = tosa.minimum %arg5, %arg6 : (tensor<32x28x35x91x53xi8>, tensor<32x28x35x1x53xi8>) -> tensor<32x28x35x91x53xi8>
    %10 = tosa.rsqrt %arg7 : (tensor<10x91x35x97xf32>) -> tensor<10x91x35x97xf32>
    %11 = tosa.reduce_product %10 {axis = 1 : i32} : (tensor<10x91x35x97xf32>) -> tensor<10x1x35x97xf32>
    %12 = tosa.greater %9, %9 : (tensor<32x28x35x91x53xi8>, tensor<32x28x35x91x53xi8>) -> tensor<32x28x35x91x53xi1>
    %13 = tosa.reduce_min %11 {axis = 0 : i32} : (tensor<10x1x35x97xf32>) -> tensor<1x1x35x97xf32>
    %14 = tosa.sub %13, %13 : (tensor<1x1x35x97xf32>, tensor<1x1x35x97xf32>) -> tensor<1x1x35x97xf32>
    %15 = tosa.argmax %10 {axis = 3 : i32} : (tensor<10x91x35x97xf32>) -> tensor<10x91x35xi32>
    %16 = tosa.floor %10 : (tensor<10x91x35x97xf32>) -> tensor<10x91x35x97xf32>
    %17 = tosa.rsqrt %16 : (tensor<10x91x35x97xf32>) -> tensor<10x91x35x97xf32>
    return %4, %6, %7, %8, %12, %14, %15, %17 : tensor<80x1x98x72xi1>, tensor<1x1x2x103087920xi1>, tensor<i1>, tensor<94x37x57x1x20x52xi1>, tensor<32x28x35x91x53xi1>, tensor<1x1x35x97xf32>, tensor<10x91x35xi32>, tensor<10x91x35x97xf32>
  }
}
