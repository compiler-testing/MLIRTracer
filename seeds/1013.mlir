module {
  func.func @main(%arg0: tensor<49x63x10xf32>, %arg1: tensor<3x2xi64>, %arg2: tensor<93x60x75x39x91x84xi1>, %arg3: tensor<93x60x1x1x91x1xi1>) -> (tensor<70x882xf32>, tensor<93x60x75x39x91x84xi1>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<6xindex>} : () -> !tosa.shape<6>
    %pad_const_0 = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<49x63x10xf32>, !tosa.shape<6>, tensor<1xf32>) -> tensor<49x63x10xf32>
    %r_1 = tosa.const_shape {values = dense<[ 35, 882 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.reshape %0, %r_1 : (tensor<49x63x10xf32>, !tosa.shape<2>) -> tensor<35x882xf32>
    %2 = tosa.concat %1, %1 {axis = 0 : i32} : (tensor<35x882xf32>, tensor<35x882xf32>) -> tensor<70x882xf32>
    %3 = tosa.rsqrt %2 : (tensor<70x882xf32>) -> tensor<70x882xf32>
    %4 = tosa.bitwise_and %arg2, %arg3 : (tensor<93x60x75x39x91x84xi1>, tensor<93x60x1x1x91x1xi1>) -> tensor<93x60x75x39x91x84xi1>
    %5 = tosa.rsqrt %3 : (tensor<70x882xf32>) -> tensor<70x882xf32>
    %6 = tosa.bitwise_not %4 : (tensor<93x60x75x39x91x84xi1>) -> tensor<93x60x75x39x91x84xi1>
    return %5, %6 : tensor<70x882xf32>, tensor<93x60x75x39x91x84xi1>
  }
}
