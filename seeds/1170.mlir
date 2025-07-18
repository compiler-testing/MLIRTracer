module {
  func.func @main(%arg0: tensor<96x88x37x75x34x31xi1>, %arg1: tensor<1x1x1x1x1x1xi1>, %arg2: tensor<5x85x56x90x44xi8>, %arg3: tensor<5x85x56x1x44xi8>, %arg4: tensor<35xi8>, %arg5: tensor<28x13x54x11x16xf32>, %arg6: tensor<45x82xi1>) -> (tensor<96x88x37x75x34x31xi1>, tensor<5x85x56x90x44xi8>, tensor<1x82xi1>, tensor<28x13x54x11x16xf32>, tensor<1xi8>, tensor<28x13x54x11x16xi1>, tensor<1x1x1x1xi8>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<96x88x37x75x34x31xi1>, tensor<1x1x1x1x1x1xi1>) -> tensor<96x88x37x75x34x31xi1>
    %1 = tosa.minimum %arg2, %arg3 : (tensor<5x85x56x90x44xi8>, tensor<5x85x56x1x44xi8>) -> tensor<5x85x56x90x44xi8>
    %2 = tosa.reduce_product %arg4 {axis = 0 : i32} : (tensor<35xi8>) -> tensor<1xi8>
    %3 = tosa.tanh %arg5 : (tensor<28x13x54x11x16xf32>) -> tensor<28x13x54x11x16xf32>
    %4 = tosa.reduce_any %arg6 {axis = 0 : i32} : (tensor<45x82xi1>) -> tensor<1x82xi1>
    %5 = tosa.abs %3 : (tensor<28x13x54x11x16xf32>) -> tensor<28x13x54x11x16xf32>
    %6 = tosa.log %5 : (tensor<28x13x54x11x16xf32>) -> tensor<28x13x54x11x16xf32>
    %7 = tosa.bitwise_not %2 : (tensor<1xi8>) -> tensor<1xi8>
    %8 = tosa.arithmetic_right_shift %2, %2 {round = false} : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %9 = tosa.equal %3, %5 : (tensor<28x13x54x11x16xf32>, tensor<28x13x54x11x16xf32>) -> tensor<28x13x54x11x16xi1>
    %r_10 = tosa.const_shape {values = dense<[ 1, 1, 1, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %10 = tosa.reshape %7, %r_10 : (tensor<1xi8>, !tosa.shape<4>) -> tensor<1x1x1x1xi8>
    return %0, %1, %4, %6, %8, %9, %10 : tensor<96x88x37x75x34x31xi1>, tensor<5x85x56x90x44xi8>, tensor<1x82xi1>, tensor<28x13x54x11x16xf32>, tensor<1xi8>, tensor<28x13x54x11x16xi1>, tensor<1x1x1x1xi8>
  }
}
