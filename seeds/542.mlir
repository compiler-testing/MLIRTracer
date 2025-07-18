module {
  func.func @main(%arg0: tensor<i8>, %arg1: tensor<i8>, %arg2: tensor<51x76x99x25x24xf32>) -> (tensor<1x1xi8>, tensor<1x1xi8>, tensor<i8>, tensor<1xi32>, tensor<51x76x99x25x24xf32>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<i8>, tensor<i8>) -> tensor<i8>
    %r_1 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.reshape %0, %r_1 : (tensor<i8>, !tosa.shape<2>) -> tensor<1x1xi8>
    %2 = tosa.maximum %1, %1 : (tensor<1x1xi8>, tensor<1x1xi8>) -> tensor<1x1xi8>
    %3 = tosa.tanh %arg2 : (tensor<51x76x99x25x24xf32>) -> tensor<51x76x99x25x24xf32>
    %4 = tosa.identity %3 : (tensor<51x76x99x25x24xf32>) -> tensor<51x76x99x25x24xf32>
    %5 = tosa.reciprocal %4 : (tensor<51x76x99x25x24xf32>) -> tensor<51x76x99x25x24xf32>
    %6 = tosa.minimum %4, %5 : (tensor<51x76x99x25x24xf32>, tensor<51x76x99x25x24xf32>) -> tensor<51x76x99x25x24xf32>
    %7 = tosa.add %6, %5 : (tensor<51x76x99x25x24xf32>, tensor<51x76x99x25x24xf32>) -> tensor<51x76x99x25x24xf32>
    %8 = tosa.abs %7 : (tensor<51x76x99x25x24xf32>) -> tensor<51x76x99x25x24xf32>
    %9 = tosa.reduce_sum %1 {axis = 0 : i32} : (tensor<1x1xi8>) -> tensor<1x1xi8>
    %10 = tosa.argmax %1 {axis = 0 : i32} : (tensor<1x1xi8>) -> tensor<1xi32>
    %11 = tosa.logical_right_shift %0, %0 : (tensor<i8>, tensor<i8>) -> tensor<i8>
    %12 = tosa.reduce_product %10 {axis = 0 : i32} : (tensor<1xi32>) -> tensor<1xi32>
    %13 = tosa.rsqrt %8 : (tensor<51x76x99x25x24xf32>) -> tensor<51x76x99x25x24xf32>
    return %2, %9, %11, %12, %13 : tensor<1x1xi8>, tensor<1x1xi8>, tensor<i8>, tensor<1xi32>, tensor<51x76x99x25x24xf32>
  }
}
