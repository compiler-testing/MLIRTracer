module {
  func.func @main(%arg0: tensor<66x85x10xf32>, %arg1: tensor<64x14x72x13x69x8xi8>, %arg2: tensor<64x1x1x13x1x1xi8>, %arg3: tensor<i1>) -> (tensor<1x85x10xf32>, tensor<64x14x72x13x69x8xi8>, tensor<1x26x552x504xi8>, tensor<128x52x1104x1008xi8>, tensor<i1>) {
    %0 = tosa.ceil %arg0 : (tensor<66x85x10xf32>) -> tensor<66x85x10xf32>
    %1 = tosa.log %0 : (tensor<66x85x10xf32>) -> tensor<66x85x10xf32>
    %2 = tosa.ceil %1 : (tensor<66x85x10xf32>) -> tensor<66x85x10xf32>
    %3 = tosa.pow %2, %1 : (tensor<66x85x10xf32>, tensor<66x85x10xf32>) -> tensor<66x85x10xf32>
    %4 = tosa.bitwise_or %arg1, %arg2 : (tensor<64x14x72x13x69x8xi8>, tensor<64x1x1x13x1x1xi8>) -> tensor<64x14x72x13x69x8xi8>
    %5 = tosa.reduce_sum %3 {axis = 0 : i32} : (tensor<66x85x10xf32>) -> tensor<1x85x10xf32>
    %6 = tosa.maximum %4, %4 : (tensor<64x14x72x13x69x8xi8>, tensor<64x14x72x13x69x8xi8>) -> tensor<64x14x72x13x69x8xi8>
    %7 = tosa.bitwise_xor %6, %6 : (tensor<64x14x72x13x69x8xi8>, tensor<64x14x72x13x69x8xi8>) -> tensor<64x14x72x13x69x8xi8>
    %r_8 = tosa.const_shape {values = dense<[ 64, 26, 552, 504 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %8 = tosa.reshape %6, %r_8 : (tensor<64x14x72x13x69x8xi8>, !tosa.shape<4>) -> tensor<64x26x552x504xi8>
    %9 = tosa.reduce_product %8 {axis = 0 : i32} : (tensor<64x26x552x504xi8>) -> tensor<1x26x552x504xi8>
    %t_10 = tosa.const_shape {values = dense<[ 2, 2, 2, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %10 = tosa.tile %8, %t_10 : (tensor<64x26x552x504xi8>, !tosa.shape<4>) -> tensor<128x52x1104x1008xi8>
    %11 = tosa.logical_not %arg3 : (tensor<i1>) -> tensor<i1>
    return %5, %7, %9, %10, %11 : tensor<1x85x10xf32>, tensor<64x14x72x13x69x8xi8>, tensor<1x26x552x504xi8>, tensor<128x52x1104x1008xi8>, tensor<i1>
  }
}
