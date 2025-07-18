module {
  func.func @main(%arg0: tensor<88x96x6x49x25x56xf32>, %arg1: tensor<1x1x1x1x25x56xf32>, %arg2: tensor<41x57x28x26xi8>) -> (tensor<88x96x6x49x25x56xf32>, tensor<88x96x6x49x25x56xf32>, tensor<88x96x6x49x25x56xf32>, tensor<41x114x28x26xi8>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<88x96x6x49x25x56xf32>, tensor<1x1x1x1x25x56xf32>) -> tensor<88x96x6x49x25x56xf32>
    %1 = tosa.clamp %0 {min_val = -6.300000e+01 : f32, max_val = 1.100000e+02 : f32} : (tensor<88x96x6x49x25x56xf32>) -> tensor<88x96x6x49x25x56xf32>
    %2 = tosa.bitwise_not %arg2 : (tensor<41x57x28x26xi8>) -> tensor<41x57x28x26xi8>
    %3 = tosa.abs %2 : (tensor<41x57x28x26xi8>) -> tensor<41x57x28x26xi8>
    %4 = tosa.identity %1 : (tensor<88x96x6x49x25x56xf32>) -> tensor<88x96x6x49x25x56xf32>
    %5 = tosa.sigmoid %4 : (tensor<88x96x6x49x25x56xf32>) -> tensor<88x96x6x49x25x56xf32>
    %6 = tosa.bitwise_xor %3, %2 : (tensor<41x57x28x26xi8>, tensor<41x57x28x26xi8>) -> tensor<41x57x28x26xi8>
    %7 = tosa.concat %6, %3 {axis = 1 : i32} : (tensor<41x57x28x26xi8>, tensor<41x57x28x26xi8>) -> tensor<41x114x28x26xi8>
    %8 = tosa.sigmoid %1 : (tensor<88x96x6x49x25x56xf32>) -> tensor<88x96x6x49x25x56xf32>
    %9 = tosa.log %0 : (tensor<88x96x6x49x25x56xf32>) -> tensor<88x96x6x49x25x56xf32>
    %10 = tosa.bitwise_not %7 : (tensor<41x114x28x26xi8>) -> tensor<41x114x28x26xi8>
    return %5, %8, %9, %10 : tensor<88x96x6x49x25x56xf32>, tensor<88x96x6x49x25x56xf32>, tensor<88x96x6x49x25x56xf32>, tensor<41x114x28x26xi8>
  }
}
