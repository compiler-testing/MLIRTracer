module {
  func.func @main(%arg0: tensor<45x52x63x49x87x62xi8>, %arg1: tensor<6x2xi32>, %arg2: tensor<81x8x58x37xi32>, %arg3: tensor<1x1x1x1xi32>, %arg4: tensor<68x25x27x4x83xi1>, %arg5: tensor<1x25x27x1x83xi1>, %arg6: tensor<90x58x72x48x59xf32>) -> (tensor<45x52x63x49x87x62xi8>, tensor<81x24x58x37xi32>, tensor<90x58x72x48x59xf32>, tensor<68x25x27x4x83xi1>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<12xindex>} : () -> !tosa.shape<12>
    %pad_const_0 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<45x52x63x49x87x62xi8>, !tosa.shape<12>, tensor<1xi8>) -> tensor<45x52x63x49x87x62xi8>
    %1 = tosa.intdiv %arg2, %arg3 : (tensor<81x8x58x37xi32>, tensor<1x1x1x1xi32>) -> tensor<81x8x58x37xi32>
    %2 = tosa.logical_and %arg4, %arg5 : (tensor<68x25x27x4x83xi1>, tensor<1x25x27x1x83xi1>) -> tensor<68x25x27x4x83xi1>
    %3 = tosa.maximum %0, %0 : (tensor<45x52x63x49x87x62xi8>, tensor<45x52x63x49x87x62xi8>) -> tensor<45x52x63x49x87x62xi8>
    %4 = tosa.reciprocal %arg6 : (tensor<90x58x72x48x59xf32>) -> tensor<90x58x72x48x59xf32>
    %t_5 = tosa.const_shape {values = dense<[ 1, 3, 1, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %5 = tosa.tile %1, %t_5 : (tensor<81x8x58x37xi32>, !tosa.shape<4>) -> tensor<81x24x58x37xi32>
    %6 = tosa.floor %4 : (tensor<90x58x72x48x59xf32>) -> tensor<90x58x72x48x59xf32>
    %7 = tosa.logical_not %2 : (tensor<68x25x27x4x83xi1>) -> tensor<68x25x27x4x83xi1>
    %8 = tosa.logical_or %7, %7 : (tensor<68x25x27x4x83xi1>, tensor<68x25x27x4x83xi1>) -> tensor<68x25x27x4x83xi1>
    return %3, %5, %6, %8 : tensor<45x52x63x49x87x62xi8>, tensor<81x24x58x37xi32>, tensor<90x58x72x48x59xf32>, tensor<68x25x27x4x83xi1>
  }
}
